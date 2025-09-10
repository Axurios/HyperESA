import torch
import math
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb
import itertools

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from typing import Optional, List
import torch_scatter

from utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP, GatedMLPMulti

from utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)

from utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder

# Task names for the LRGB peptides-func benchmark
pept_struct_target_names = ["Inertia_mass_a", "Inertia_mass_b", "Inertia_mass_c",
                        "Inertia_valence_a", "Inertia_valence_b",
                        "Inertia_valence_c", "length_a", "length_b", "length_c",
                        "Spherocity", "Plane_best_fit"]


def nearest_multiple_of_8(n):
    return math.ceil(n / 8) * 8



class Estimator(pl.LightningModule):
    def __init__(
        self,
        task_type: str,
        num_features: int,
        graph_dim: int,
        edge_dim: int,
        batch_size: int = 32,
        lr: float = 0.001,
        linear_output_size: int = 1,
        scaler=None,
        monitor_loss_name: str = "val_loss",
        xformers_or_torch_attn: str = "xformers",
        hidden_dims: List[int] = None,
        num_heads: int = None,
        num_sabs: int = None,
        sab_dropout: float = 0.0,
        mab_dropout: float = 0.0,
        pma_dropout: float = 0.0,
        apply_attention_on: str = "edge",
        layer_types: List[str] = None,
        use_mlps: bool = False,
        set_max_items: int=0,
        regression_loss_fn: str = "mae",
        early_stopping_patience: int = 30,
        optimiser_weight_decay: float = 1e-3,
        mlp_hidden_size: int = 64,
        mlp_type: str = "standard",
        attn_residual_dropout: float = 0.0,
        norm_type: str = "LN",
        triu_attn_mask: bool = False,
        output_save_dir: str = None,
        use_bfloat16: bool = True,
        is_node_task: bool = False,
        train_mask = None,
        val_mask = None,
        test_mask = None,
        posenc: str = None,
        num_mlp_layers: int = 3,
        pre_or_post: str = "pre",
        pma_residual_dropout: float = 0,
        use_mlp_ln: bool = False,
        mlp_dropout: float = 0,
        **kwargs,
    ):
        super().__init__()
        assert task_type in ["binary_classification", "multi_classification", "regression"]

        self.graph_dim = graph_dim
        self.task_type = task_type

        # print(task_type) # prints regression ok

        self.edge_dim = edge_dim

        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = scaler


        self.target_names = kwargs.get("dataset_target_name", None)
        if not isinstance(self.target_names, list):
            self.target_names = [self.target_names]
        # print("target names in estimator: ", self.target_names)

        linear_output_size = len(self.target_names) if self.target_names else 1
        self.linear_output_size = linear_output_size
        #print("linear output size: ", self.linear_output_size)
        #print("target names: ", self.target_names)


        self.monitor_loss_name = monitor_loss_name
        self.mlp_hidden_size = mlp_hidden_size
        self.norm_type = norm_type
        self.set_max_items = set_max_items
        self.output_save_dir = output_save_dir
        self.is_node_task = is_node_task
        self.use_mlp_ln = use_mlp_ln
        self.pre_or_post = pre_or_post
        self.mlp_dropout = mlp_dropout

        self.use_mlps = use_mlps
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay
        self.regression_loss_fn = regression_loss_fn
        self.mlp_type = mlp_type
        self.attn_residual_dropout = attn_residual_dropout
        self.pma_residual_dropout = pma_residual_dropout
        self.triu_attn_mask = triu_attn_mask
        self.use_bfloat16 = use_bfloat16
        self.layer_types = layer_types

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.val_test_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.val_preds = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.val_test_metrics = {}
        self.test_metrics = {}

        # Holds final graphs embeddings
        self.test_graph_embeddings = defaultdict(list)
        self.val_graph_embeddings = defaultdict(list)
        self.train_graph_embeddings = defaultdict(list)

        self.xformers_or_torch_attn = xformers_or_torch_attn
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.num_sabs = num_sabs
        self.sab_dropout = sab_dropout
        self.mab_dropout = mab_dropout
        self.pma_dropout = pma_dropout
        self.apply_attention_on = apply_attention_on
        self.posenc = posenc
        self.num_mlp_layers = num_mlp_layers
        self.use_hyperedges = kwargs.get('use_hyperedges', False)

        self.rwse_encoder = None
        self.lap_encoder = None

        if "RWSE" in self.posenc:
            self.rwse_encoder = KernelPENodeEncoder()
        if "LapPE" in self.posenc:
            self.lap_encoder = LapPENodeEncoder()
 
        if self.norm_type == "BN":
            norm_fn = BN
        elif self.norm_type == "LN":
            norm_fn = LN

        if self.apply_attention_on == "node":
            in_dim = self.num_features
            if self.rwse_encoder is not None:
                in_dim += 24
            if self. lap_encoder is not None:
                in_dim += 4
            
            if self.mlp_type in ["standard", "gated_mlp"]:
                self.node_mlp = SmallMLP(
                    in_dim=in_dim,
                    inter_dim=128,
                    out_dim=self.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=self.num_mlp_layers if self.num_mlp_layers > 1 else self.num_mlp_layers + 1,
                )

            # Uncomment if you want the gated MLP here
            # elif self.mlp_type == "gated_mlp":
            #     self.node_mlp = GatedMLPMulti(
            #         in_dim=in_dim,
            #         out_dim=self.hidden_dims[0],
            #         inter_dim=128,
            #         activation=F.silu,
            #         dropout_p=0,
            #         num_layers=self.num_mlp_layers,
            #     )


        elif self.apply_attention_on == "edge":
            in_dim = self.num_features
            if self.rwse_encoder is not None:
                in_dim += 24
            if self.lap_encoder is not None:
                in_dim += 4

            in_dim = in_dim * 2
            if self.edge_dim is not None:
                in_dim += self.edge_dim
            
            if self.mlp_type in ["standard", "gated_mlp"]:

                print(f"output dim of node/edge MLP: {self.hidden_dims[0]}")

                self.node_edge_mlp = SmallMLP(
                    in_dim=in_dim,
                    inter_dim=128,
                    out_dim=self.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=self.num_mlp_layers,
                )

            # Uncomment if you want the gated MLP here

            # elif self.mlp_type == "gated_mlp":
            #     self.node_edge_mlp = GatedMLPMulti(
            #         in_dim=in_dim,
            #         out_dim=self.hidden_dims[0],
            #         inter_dim=128,
            #         activation=F.silu,
            #         dropout_p=0,
            #         num_layers=self.num_mlp_layers,
            #     )
            

        self.mlp_norm = norm_fn(self.hidden_dims[0])

        st_args = dict(
            num_outputs=32, # this is the k for PMA
            dim_output=self.graph_dim,
            xformers_or_torch_attn=self.xformers_or_torch_attn,
            dim_hidden=self.hidden_dims,
            num_heads=self.num_heads,
            sab_dropout=self.sab_dropout,
            mab_dropout=self.mab_dropout,
            pma_dropout=self.pma_dropout,
            use_mlps=self.use_mlps,
            mlp_hidden_size=self.mlp_hidden_size,
            mlp_type=self.mlp_type,
            norm_type=self.norm_type,
            node_or_edge=self.apply_attention_on,
            residual_dropout=self.attn_residual_dropout,
            set_max_items=nearest_multiple_of_8(self.set_max_items + 1),
            use_bfloat16=self.use_bfloat16,
            layer_types=self.layer_types,
            num_mlp_layers=self.num_mlp_layers,
            pre_or_post=self.pre_or_post,
            pma_residual_dropout=self.pma_residual_dropout,
            use_mlp_ln=self.use_mlp_ln,
            mlp_dropout=self.mlp_dropout,
        )

        self.st_fast = ESA(**st_args)

        #print(self.mlp_type)
        if self.mlp_type in ["standard", "gated_mlp"]:
            #print(f"activated mlp type: {self.mlp_type}")

            self.output_mlp = SmallMLP(
                in_dim= self.graph_dim, #AD
                inter_dim=128,
                out_dim=self.linear_output_size,
                use_ln=False,
                dropout_p=0,
                num_layers=self.num_mlp_layers if self.num_mlp_layers > 1 else self.num_mlp_layers + 1,
            )
            #AD
            self.output_mlp = nn.Linear(self.graph_dim, self.linear_output_size)

        # Uncomment if you want the gated MLP here
            
        # elif self.mlp_type == "gated_mlp":
        #     self.output_mlp = GatedMLPMulti(
        #         in_dim=self.graph_dim,
        #         out_dim=self.linear_output_size,
        #         inter_dim=128,
        #         activation=F.silu,
        #         dropout_p=0,
        #         num_layers=self.num_mlp_layers,
        #     )


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mapping: torch.Tensor,
        edge_attr: torch.Tensor,
        num_max_items: int,
        batch,
    ):
        
        x = x.float()

        if self.lap_encoder is not None:
            lap_pos_enc = self.lap_encoder(batch.EigVals, batch.EigVecs)
            x = torch.cat((x, lap_pos_enc), 1)
        if self.rwse_encoder is not None:
            rwse_pos_enc = self.rwse_encoder(batch.pestat_RWSE)
            x = torch.cat((x, rwse_pos_enc), 1)

        # ESA
        if self.apply_attention_on == "edge":
            source = x[edge_index[0, :], :] ; target = x[edge_index[1, :], :]
            h = torch.cat((source, target), dim=1)

            if self.edge_dim is not None and edge_attr is not None:
                h = torch.cat((h, edge_attr.float()), dim=1)
            edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])

            # print("there")
            h_hyperedge_output = None
            if self.use_hyperedges and hasattr(batch, 'hyperedges') and batch.hyperedges:
                ptr = batch.ptr.to(x.device)
                all_hyperedges = [torch.tensor(hed, dtype=torch.long, device=x.device)  + ptr[g] for g, gra in enumerate(batch.hyperedges) for hed in gra]

                hedge_nodes_tensor = torch.nn.utils.rnn.pad_sequence(all_hyperedges, batch_first=True, padding_value=-1)
                mask = hedge_nodes_tensor != -1 ; valid_indices = hedge_nodes_tensor.clone() ; valid_indices[~mask] = 0 
                features = x[valid_indices] ; features = features * mask.unsqueeze(-1)

                h_x_mean = features.sum(dim=1) / mask.sum(dim=1, keepdim=True)
                h_hyperedge_output = torch.cat((h_x_mean, h_x_mean), dim=1)

                num_hyperedges_per_graph = [len(h) for h in batch.hyperedges] #; print("HYPEREDGES USED")
                hedge_index = torch.repeat_interleave(torch.arange(len(batch.hyperedges), device=x.device), torch.tensor(num_hyperedges_per_graph, device=x.device))
                
            
                # The final shape is [num_hyperedges, 2*n + e]
                if self.edge_dim is not None and edge_attr is not None:   
                    total_hyperedges = sum(len(graph_hyperedges) for graph_hyperedges in batch.hyperedges) 
                    hyperedge_attr_ones = torch.ones((total_hyperedges, self.edge_dim), device=x.device, dtype=x.dtype) # this might be the issue
                    h_hyperedge_output = torch.cat((h_hyperedge_output, hyperedge_attr_ones), dim=1)
                

                if h_hyperedge_output is not None:
                    h_combined = torch.cat((h, h_hyperedge_output), dim=0)
                    edge_batch_index = batch_mapping.index_select(0, edge_index[0, :])
                    combined_batch_index = torch.cat((edge_batch_index, hedge_index), dim=0)
                    h = h_combined ; edge_batch_index = combined_batch_index


            h = self.node_edge_mlp(h)
            h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
            # edge_index unchanged, not the same as h dimension ???
            h = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)


        # NSA
        else:
            h = self.mlp_norm(self.node_mlp(x))

            h, dense_batch_index = to_dense_batch(h, batch_mapping, fill_value=0, max_num_nodes=num_max_items)
            h = self.st_fast(h, edge_index, batch_mapping, num_max_items=num_max_items)

            if self.is_node_task:
                h = h[dense_batch_index]

        #predictions = torch.flatten(self.output_mlp(torch.flatten(h, start_dim=1)))
        predictions = self.output_mlp(torch.flatten(h, start_dim=1))
        return predictions
    

    def configure_optimizers(self):
        opt = bnb.optim.AdamW8bit(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)

        self.monitor_loss_name = "Validation MCC" if "MCC" in self.monitor_loss_name or self.monitor_loss_name == "MCC" else self.monitor_loss_name
        mode = "max" if "MCC" in self.monitor_loss_name else "min"

        opt_dict = {
            "optimizer": opt,
            "monitor": self.monitor_loss_name,
        }

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=mode, factor=0.5, patience=self.early_stopping_patience // 2, verbose=True
        )
        if self.monitor_loss_name != "train_loss":
            opt_dict["lr_scheduler"] = sched

        return opt_dict
    

    def _batch_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: Optional[torch.Tensor],
        batch_mapping: Optional[torch.Tensor],
        num_max_items: int,
        edge_attr: Optional[torch.Tensor] = None,
        step_type: Optional[str] = None,
        batch = None,
    ):
        predictions = self.forward(x, edge_index, batch_mapping, edge_attr=edge_attr, num_max_items=num_max_items, batch=batch)


        if self.task_type == "multi_classification":
            predictions = predictions.reshape(-1, self.linear_output_size)

            # Graph-level task
            if self.train_mask is None:
                task_loss = F.cross_entropy(predictions.squeeze().float(), y.squeeze().long())
            # Node-level task
            else:
                predictions = predictions.squeeze().float()
                y = y.squeeze().long()

                if step_type == "train" and self.train_mask is not None:
                    predictions = predictions[self.train_mask]
                    y = y[self.train_mask]
                
                if step_type == "validation" and self.val_mask is not None:
                    predictions = predictions[self.val_mask]
                    y = y[self.val_mask]

                if "test" in step_type and self.test_mask is not None:
                    predictions = predictions[self.test_mask]
                    y = y[self.test_mask]

                task_loss = F.cross_entropy(predictions, y)
            per_target_losses = None
            total_loss = task_loss

        elif self.task_type == "binary_classification":
            y = y.view(predictions.shape)
            task_loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())
            per_target_losses = None
            total_loss = task_loss

        # else:
        #     if self.regression_loss_fn == "mse":
        #         task_loss = F.mse_loss(torch.flatten(predictions), torch.flatten(y.float()))
        #     elif self.regression_loss_fn == "mae":
        #         task_loss = F.l1_loss(predictions, y.float())
        else:
            # Use reduction='none' to get the loss for each individual element
            if self.regression_loss_fn == "mse":
                elementwise_loss = F.mse_loss(predictions, y.float(), reduction='none')
            elif self.regression_loss_fn == "mae":
                elementwise_loss = F.l1_loss(predictions, y.float(), reduction='none')
                
            # per_target_losses will have shape [num_targets]
            # We take the mean loss for each target over the entire batch
            per_target_losses = elementwise_loss.mean(dim=0)

            # total_loss is the mean of all per-target losses
            total_loss = per_target_losses.mean()

        return total_loss, per_target_losses, predictions, y
        #return task_loss, predictions, y


    def _step(self, batch: torch.Tensor, step_type: str):
        assert step_type in ["train", "validation", "test", "validation_test"]

        x, edge_index, y, batch_mapping, edge_attr = batch.x, batch.edge_index, batch.y, batch.batch, batch.edge_attr
        max_node, max_edge = batch.max_node_global, batch.max_edge_global

        if self.apply_attention_on == "edge":
            num_max_items = max_edge
        else:
            num_max_items = max_node

        num_max_items = torch.max(num_max_items).item()
        num_max_items = nearest_multiple_of_8(num_max_items + 1)

        task_loss, per_target_losses, predictions, y = self._batch_loss(
            x, edge_index, y, batch_mapping, edge_attr=edge_attr, num_max_items=num_max_items, step_type=step_type, batch=batch
        )

        # --- Logging the individual losses for regression ---
        if self.task_type == "regression" and self.linear_output_size > 1:
            for idx, target_name in enumerate(self.target_names):
                loss = per_target_losses[idx]
                self.log(
                    f"{step_type}_loss_{target_name}", 
                    loss, 
                    prog_bar=False, 
                    batch_size=self.batch_size
                )

        predictions = predictions.detach().squeeze()

        if self.task_type == "regression":
            output = (predictions.cpu(), y.cpu())
        elif "classification" in self.task_type:
            output = (predictions.cpu(), y.cpu())

        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "validation_test":
            self.val_test_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)

        return task_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_total_loss = self._step(batch, "train")

        if train_total_loss:
            self.log("train_loss", train_total_loss, prog_bar=True)

        return train_total_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            val_total_loss = self._step(batch, "validation")

            self.log("val_loss", val_total_loss)

            return val_total_loss

        if dataloader_idx == 1:
            val_test_total_loss = self._step(batch, "validation_test")

            self.log("val_test_loss", val_test_total_loss)

            return val_test_total_loss


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_total_loss = self._step(batch, "test")

        self.log("test_loss", test_total_loss)

        return test_total_loss


    def _epoch_end_report(self, epoch_outputs, epoch_type):
        # The most robust way to combine tensors, regardless of batch_size
        # or whether the output is single-target or multi-target
        y_pred = torch.cat([item[0] for item in epoch_outputs], dim=0)
        y_true = torch.cat([item[1] for item in epoch_outputs], dim=0)

        # y_pred and y_true should now have shape [total_samples, linear_output_size]
        
        # We need to make sure the scaler logic is also using the right shape
        if self.scaler:
            if self.linear_output_size > 1:
                y_pred = self.scaler.inverse_transform(y_pred.cpu().reshape(-1, self.linear_output_size))
                y_true = self.scaler.inverse_transform(y_true.cpu().reshape(-1, self.linear_output_size))
            else:
                y_pred = self.scaler.inverse_transform(y_pred.cpu().reshape(-1, 1)).flatten()
                y_true = self.scaler.inverse_transform(y_true.cpu().reshape(-1, 1)).flatten()
            
            y_pred = torch.from_numpy(y_pred).to(self.device)
            y_true = torch.from_numpy(y_true).to(self.device)

        # Now, the rest of the logic can operate on correctly shaped tensors
        if self.task_type == "binary_classification" and self.linear_output_size > 1:
            y_true = y_true.detach().cpu().reshape(-1, self.linear_output_size).long()
            y_pred = y_pred.detach().cpu().reshape(-1, self.linear_output_size)
            metrics = get_cls_metrics_multilabel_pt(y_true, y_pred, self.linear_output_size)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "binary_classification" and self.linear_output_size == 1:
            metrics = get_cls_metrics_binary_pt(y_true, y_pred)
            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        elif self.task_type == "multi_classification" and self.linear_output_size > 1:
            metrics = get_cls_metrics_multiclass_pt(y_true, y_pred, self.linear_output_size)
            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)
            self.log(f"{epoch_type} AP", metrics[4], batch_size=self.batch_size)

        elif self.task_type == "regression":
            if self.linear_output_size > 1 and self.target_names is not None:
                # Loop over each target name and its corresponding column index
                for idx, target_name in enumerate(self.target_names):
                    # Extract predictions and true values for this specific target
                    # No squeeze() is needed here since y_pred/y_true are already 2D
                    target_y_pred = y_pred[:, idx]
                    target_y_true = y_true[:, idx]
                    
                    # Calculate metrics for this single target
                    metrics = get_regr_metrics_pt(target_y_true, target_y_pred)

                    # Log the individual metrics
                    self.log(f"{epoch_type} {target_name} R2", metrics["R2"], batch_size=self.batch_size)
                    self.log(f"{epoch_type} {target_name} MAE", metrics["MAE"], batch_size=self.batch_size)
                    self.log(f"{epoch_type} {target_name} RMSE", metrics["RMSE"], batch_size=self.batch_size)
                    self.log(f"{epoch_type} {target_name} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)
                
                # Calculate and log the average MAE across all targets
                mae_values = [get_regr_metrics_pt(y_true[:, idx], y_pred[:, idx])["MAE"].cpu() for idx in range(self.linear_output_size)]
                mae_avg = np.mean(np.array(mae_values))
                self.log(f"{epoch_type} AVERAGE MAE", mae_avg, batch_size=self.batch_size)
            
            else:
                metrics = get_regr_metrics_pt(y_true.squeeze(), y_pred.squeeze())
                self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
                self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
                self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
                self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)
        
        return metrics, y_pred, y_true
    

        # def flatten_list_of_tensors(lst):
        #     try:
        #         return np.array([item.item() for sublist in lst for item in sublist])
        #     except:
        #         return torch.cat(lst, dim=0)

        # if self.task_type == "regression":
        #     y_pred = flatten_list_of_tensors([item[0] for item in epoch_outputs]).reshape(-1, self.linear_output_size)
        #     y_true = flatten_list_of_tensors([item[1] for item in epoch_outputs]).reshape(-1, self.linear_output_size)
        # else:
        #     if self.batch_size > 1:
        #         y_pred = torch.cat([item[0] for item in epoch_outputs], dim=0)
        #         y_true = torch.cat([item[1] for item in epoch_outputs], dim=0)
        #     else:
        #         y_pred = torch.cat([item[0].unsqueeze(0) for item in epoch_outputs], dim=0).squeeze()
        #         y_true = torch.cat([item[1].unsqueeze(0) for item in epoch_outputs], dim=0).squeeze()

        # if self.scaler:
        #     if self.linear_output_size > 1:
        #         y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
        #         y_true = self.scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
        #     else:
        #         y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        #         y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

        #     y_pred = torch.from_numpy(y_pred)
        #     y_true = torch.from_numpy(y_true)

        # if self.task_type == "binary_classification" and self.linear_output_size > 1:
        #     y_true = y_true.detach().cpu().reshape(-1, self.linear_output_size).long()
        #     y_pred = y_pred.detach().cpu().reshape(-1, self.linear_output_size)

        #     metrics = get_cls_metrics_multilabel_pt(y_true, y_pred, self.linear_output_size)

        #     self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        # elif self.task_type == "binary_classification" and self.linear_output_size == 1:
        #     metrics = get_cls_metrics_binary_pt(y_true, y_pred)

        #     self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)

        # elif self.task_type == "multi_classification" and self.linear_output_size > 1:
        #     metrics = get_cls_metrics_multiclass_pt(y_true, y_pred, self.linear_output_size)

        #     self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} MCC", metrics[1], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} Accuracy", metrics[2], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} F1", metrics[3], batch_size=self.batch_size)
        #     self.log(f"{epoch_type} AP", metrics[4], batch_size=self.batch_size)

        # elif self.task_type == "regression":
        #     if self.linear_output_size > 1 and self.target_names is not None:
        #     # --- START of changed code ---
        #     # Loop over each target name and its corresponding column index
        #         for idx, target_name in enumerate(self.target_names):
                    
        #             # Extract predictions and true values for this specific target
        #             target_y_pred = y_pred.squeeze()[:, idx]
        #             target_y_true = y_true.squeeze()[:, idx]
                    
        #             # Calculate metrics for this single target
        #             metrics = get_regr_metrics_pt(target_y_true, target_y_pred)

        #             # Log the individual metrics
        #             self.log(f"{epoch_type} {target_name} R2", metrics["R2"], batch_size=self.batch_size)
        #             self.log(f"{epoch_type} {target_name} MAE", metrics["MAE"], batch_size=self.batch_size)
        #             self.log(f"{epoch_type} {target_name} RMSE", metrics["RMSE"], batch_size=self.batch_size)
        #             self.log(f"{epoch_type} {target_name} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)
                
        #         # Calculate and log the average MAE across all targets
        #         mae_values = [get_regr_metrics_pt(y_true.squeeze()[:, idx], y_pred.squeeze()[:, idx])["MAE"] for idx in range(self.linear_output_size)]
        #         mae_avg = np.mean(np.array(mae_values))
        #         self.log(f"{epoch_type} AVERAGE MAE", mae_avg, batch_size=self.batch_size)
        #         # --- END of changed code ---
            
        #     # if self.linear_output_size != 11:
        #     #     metrics = get_regr_metrics_pt(y_true.squeeze(), y_pred.squeeze())

        #     #     self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
        #     #     self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
        #     #     self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
        #     #     self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)
        #     else:
        #         metrics = get_regr_metrics_pt(y_true.squeeze(), y_pred.squeeze())
        #         self.log(f"{epoch_type} R2", metrics["R2"], batch_size=self.batch_size)
        #         self.log(f"{epoch_type} MAE", metrics["MAE"], batch_size=self.batch_size)
        #         self.log(f"{epoch_type} RMSE", metrics["RMSE"], batch_size=self.batch_size)
        #         self.log(f"{epoch_type} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)
        #         # mae_avg = []
        #         # for idx in range(11):
        #         #     target_name = pept_struct_target_names[idx]
        #         #     metrics = get_regr_metrics_pt(y_true.squeeze()[:, idx], y_pred.squeeze()[:, idx])

        #         #     mae_avg.append(metrics["MAE"])

        #         #     self.log(f"{epoch_type} {target_name} R2", metrics["R2"], batch_size=self.batch_size)
        #         #     self.log(f"{epoch_type} {target_name} MAE", metrics["MAE"], batch_size=self.batch_size)
        #         #     self.log(f"{epoch_type} {target_name} RMSE", metrics["RMSE"], batch_size=self.batch_size)
        #         #     self.log(f"{epoch_type} {target_name} SMAPE", metrics["SMAPE"], batch_size=self.batch_size)

        #         # mae_avg = np.mean(np.array(mae_avg))
        #         # self.log(f"{epoch_type} AVERAGE MAE", mae_avg, batch_size=self.batch_size)
        # return metrics, y_pred, y_true


    def on_train_epoch_end(self):
        self.train_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
            self.train_output[self.current_epoch], epoch_type="Train"
        )

        del y_pred
        del y_true
        del self.train_output[self.current_epoch]


    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            self.val_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_output[self.current_epoch], epoch_type="Validation"
            )

            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

        if len(self.val_test_output[self.current_epoch]) > 0:
            self.val_test_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_test_output[self.current_epoch], epoch_type="ValidationTest"
            )

            del y_pred
            del y_true
            del self.val_test_output[self.current_epoch]


    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        self.test_metrics[self.num_called_test], y_pred, y_true = self._epoch_end_report(
            test_outputs_per_epoch, epoch_type="Test"
        )
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true

        self.num_called_test += 1





# def find_max_robust(nested_list):
#     """
#     Finds the maximum integer in a nested list of any shape.
#     This is a more robust solution than using fixed nested loops.

#     Args:
#         nested_list (list): The list containing integers and/or other lists.

#     Returns:
#         int or None: The maximum integer found, or None if the list is empty.
#     """
#     # Use a list as a stack for an iterative, non-recursive approach
#     stack = [nested_list]
#     max_value = float('-inf')
#     found_number = False

#     while stack:
#         current_item = stack.pop()

#         if isinstance(current_item, list):
#             # If the item is a list, extend the stack with its contents
#             stack.extend(current_item)
#         elif isinstance(current_item, (int, float)):
#             # If the item is a number, compare and update the maximum
#             if current_item > max_value:
#                 max_value = current_item
#             found_number = True
#         # Ignore other data types for this specific task

#     if not found_number:
#         return None
        
#     return max_value


# def count_unique_integers(nested_list):
#     """
#     Counts the number of unique integers in a nested list of any shape.

#     Args:
#         nested_list (list): The list containing integers and/or other lists.

#     Returns:
#         int: The number of different integers found.
#     """
#     # Use a set to store unique integers.
#     unique_integers = set()

#     # Use a list as a stack for an iterative, non-recursive approach
#     stack = [nested_list]

#     while stack:
#         current_item = stack.pop()

#         if isinstance(current_item, list):
#             # If the item is a list, extend the stack with its contents
#             stack.extend(current_item)
#         elif isinstance(current_item, int):
#             # If the item is an integer, add it to the set.
#             # The set automatically handles uniqueness.
#             unique_integers.add(current_item)
#         # Note: We are specifically looking for integers, so we will not add floats.
#         # This can be easily changed by adding `float` to the `isinstance` check.

#     print(f"The unique integers found are: {sorted(list(unique_integers))}")
#     return len(unique_integers)

            # # ----------------- HYPEREDGES -----------------
            # # h_hyperedge_output = None
            # # if self.use_hyperedges and hasattr(batch, 'hyperedges') and batch.hyperedges:
                
            #     if total_hyperedges == 0:
            #         print("no hyperedges ???")
            #         pass
            #     total_hyperedges = sum(len(graph_hyperedges) for graph_hyperedges in batch.hyperedges)
            #     hyperedge_nodes_shifted = [] ; hyperedge_index_map = []
            #     hyperedge_nodes_shifted_list = [] ; hyperedge_index_map_list = []
            
            #     num_hyperedges_per_graph = torch.tensor([len(g_h) for g_h in batch.hyperedges], device=x.device, dtype=torch.long)
            #     hyperedge_id_shifts = torch.cumsum(num_hyperedges_per_graph, dim=0)
            #     hyperedge_id_shifts = torch.cat([torch.tensor([0], device=x.device), hyperedge_id_shifts[:-1]])
            #     for i in range(len(batch.hyperedges)):
            #         graph_hyperedges = batch.hyperedges[i] ; node_shift = batch.ptr[i]
            #         if not graph_hyperedges:
            #             continue
            #         flat_nodes = torch.tensor([node for h_nodes in graph_hyperedges for node in h_nodes], device=x.device, dtype=torch.long)
            #         shifted_nodes = flat_nodes + node_shift
            #         hyperedge_nodes_shifted_list.append(shifted_nodes)
                    
            #         hyperedge_lengths = torch.tensor([len(h_nodes) for h_nodes in graph_hyperedges], device=x.device, dtype=torch.long)
            #         hyperedge_ids = torch.arange(len(graph_hyperedges), device=x.device, dtype=torch.long) + hyperedge_id_shifts[i]
            #         hyperedge_index_map_list.append(torch.repeat_interleave(hyperedge_ids, hyperedge_lengths))
                    
            #     hyperedge_nodes_shifted = torch.cat(hyperedge_nodes_shifted_list, dim=0)
            #     hyperedge_index_map = torch.cat(hyperedge_index_map_list, dim=0)

            #     h_x_mean = torch_scatter.scatter_mean(x[hyperedge_nodes_shifted], hyperedge_index_map, dim=0)  # Now perform the scatter operation with the correctly shifted indices


                # print("Type of all_hyperedges:", type(all_hyperedges)) # list
                # print("Type of all_hyperedges[0]:", type(all_hyperedges[0])) # tensor
                # print(len(all_hyperedges), "length of all_hyperedges list") # (should be Eh, the total number of hyperedges in the batch)
                # print(all_hyperedges[0].shape)
                # print(all_hyperedges[1].shape) 





                    # hyperedge_batch_index = torch.cat([ 
                    #     torch.full((len(graph_hyperedge),), i, device=x.device, dtype=torch.long)
                    #     for i, graph_hyperedge in enumerate(batch.hyperedges)
                    # ])