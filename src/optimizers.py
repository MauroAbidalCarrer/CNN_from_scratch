from datetime import datetime
from functools import reduce
from itertools import accumulate
from IPython.display import display
from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame as DF
from plotly.express import scatter
from plotly.graph_objects import FigureWidget
from numpy import ndarray, array_split as ndarray, split

from losses import Loss
from layers import Layer
from constants import EPSILON, MAX_NB_SAMPLES
from time_utils import time_to_exec
from numpy_utils import cached_zeros
from metrics import metric_func, accuracy


@dataclass
class Adam:
    nn:list[Layer]
    x:ndarray
    y:ndarray
    loss:Loss
    starting_lr:float
    lr_decay:float
    momentum_weight:float
    ada_grad_weight:float
    training_metrics:list[dict] = field(default_factory=list, init=False)
    epoch: int = field(default=0, init=False)
    iteration: int = field(default=0, init=False)

    def optimize_nn(self, epochs, batch_size, metric_freq=1, metrics:list[metric_func]=[accuracy], catch_interrupt=True, plt_x=None, plt_ys=None, **plt_kwargs) -> DF:
        """Optimizes the neural network and returns a dataframe of the training metrics."""
        if catch_interrupt:
            try:
                self._optimize_nn(epochs, batch_size, metric_freq, metrics, plt_x, plt_ys, **plt_kwargs)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt exception, returning training metrics.")
        else:
            self._optimize_nn(epochs, batch_size, metric_freq, metrics, plt_x, plt_ys, **plt_kwargs)
        return DF.from_records(self.training_metrics)

    def _optimize_nn(self, epochs, batch_size, metric_freq=1, metrics:list[metric_func]=[accuracy], plt_x=None, plt_ys=None, **plt_kwargs) -> DF:
        nb_batches = int(np.ceil(len(self.x) / batch_size))
        fig = None
        # Use self.epoch instead of epoch.
        # This avoids resetting new metrics DF lines to the same epoch value in case this method gets recalled.
        for _ in range(epochs):
            if self.epoch % metric_freq == 0:
                with time_to_exec("metric recording"):
                    self.record_metrics(metrics)
                    if not plt_x is None:
                        fig = self.create_figure_widget(plt_x, plt_ys, **plt_kwargs) if fig is None else fig
                        self.update_figure(fig, plt_x, plt_ys)
            # Shuffle x and y
            with time_to_exec("steps performing"):
                permutation = np.random.permutation(len(self.x))
                self.x = self.x[permutation]
                self.y = self.y[permutation]
                for batch_x, batch_y in zip(split(self.x, nb_batches), split(self.y, nb_batches)):
                    self.step(batch_x, batch_y)
            self.epoch += 1

    def record_metrics(self, metric_funcs:list[callable]) -> dict[str, any]:
        y_preds = []
        for subset_i in range(0, self.x.shape[0], MAX_NB_SAMPLES):
            activations = self.forward(self.x[subset_i:subset_i+MAX_NB_SAMPLES])
            y_preds.append(activations[-1])
        y_pred = np.concatenate(y_preds)
        metric_kwargs = dict(nn=self.nn, activations=activations, y_pred=y_pred, y_true=self.y, loss=self.loss)
        new_metric_line = reduce(
            lambda metric_line, metric_func: metric_func(metric_line, **metric_kwargs),
            metric_funcs,
            {
                "iteration": self.iteration,
                "epoch": self.epoch,
                "loss": self.loss.forward(y_pred, self.y),
                "learning_rate": self.learning_rate,
                "time": datetime.now()
            }
        )
        self.training_metrics.append(new_metric_line)

    def create_figure_widget(self, plt_x:str, plt_ys:list[str], **plt_kwargs) -> FigureWidget:
        df = DF.from_records(self.training_metrics).melt(plt_x, plt_ys)
        fig = scatter(df, plt_x, "value", facet_row="variable", color="variable", **plt_kwargs).update_yaxes(matches=None)
        fig = FigureWidget(fig)
        display(fig)
        return fig

    def update_figure(self, fig:FigureWidget, plt_x:str, plt_ys:list[str]):
        df = DF.from_records(self.training_metrics)
        with fig.batch_update():
            for i, plt_y in enumerate(plt_ys):
                fig.data[i].x = df[plt_x]
                fig.data[i].y = df[plt_y]

    def step(self, batch_x:ndarray, batch_y:ndarray):
        batch_activations = self.forward(batch_x)
        batch_y_preds = batch_activations[-1]
        batch_gradients = self.loss.backward(batch_y_preds, batch_y)
        reduce(self.update_layer_params, reversed(self.nn), batch_gradients)
        self.iteration += 1

    def update_layer_params(self, outputs_gradients:ndarray, layer:Layer) -> ndarray:
        grads_wrt:dict = layer.backward(outputs_gradients)
        grads_wrt_inputs = grads_wrt.pop("inputs")
        for param_name, grad_wrt_param in grads_wrt.items():
            param:ndarray = getattr(layer, param_name)
            grad_wrt_param = self.postprocess_gradient(layer, param_name, grad_wrt_param)
            setattr(layer, param_name, param - grad_wrt_param)
        return grads_wrt_inputs

    def postprocess_gradient(self, layer:Layer, param_name:str, gradient_wrt_param:ndarray) -> ndarray:
        momentum = self.update_moving_average(layer, param_name + "_momentum", gradient_wrt_param, self.momentum_weight)
        cache = self.update_moving_average(layer, param_name + "_cache", gradient_wrt_param ** 2, self.ada_grad_weight)
        return self.learning_rate * momentum / (np.sqrt(cache) + EPSILON)

    def update_moving_average(self, layer, moving_average_name:str, gradient:ndarray, beta:float) -> ndarray:
        """Updates the moving average and returns it corrected."""
        moving_average = getattr(layer, moving_average_name, cached_zeros(gradient.shape)) # Get the moving averate
        moving_average = lerp(moving_average, gradient, beta) # Move the average
        setattr(layer, moving_average_name, moving_average) # Set the moved average for future use
        return moving_average / (1 - beta ** (self.iteration + 1)) # Correct and return the average 

    @property
    def learning_rate(self) -> float: 
        return self.starting_lr / (1 + self.lr_decay * self.iteration)

    # Forward is implemented in SGD because it's the only place it's used, eventually it will be in a NueralNetwork module.
    def forward(self, inputs:ndarray) -> list[ndarray]:
        return list(accumulate(self.nn, lambda x, l: l.forward(x), initial=inputs))

def lerp(a:ndarray, b:ndarray, t:float) -> ndarray:
    return a * t + (1 - t) * b