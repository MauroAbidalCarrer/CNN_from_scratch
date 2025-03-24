from functools import reduce, cache
from IPython.display import display

import numpy as np
from plotly.graph_objects import FigureWidget
# from rich.progress import track
from plotly.express import scatter
from pandas import DataFrame as DF
from numpy import ndarray, array_split

from losses import Loss
from layers import Layer
from metrics import accuracy
from constants import EPSILON


layers = list[Layer]

class Adam:
    def __init__(self, starting_lr:float, lr_decay:float, momentum_weight:float, ada_grad_weight:float):
        self.iterations = 0
        self.training_metrics:list[dict] = []
        self.momentum_weight = momentum_weight
        self.ada_grad_weight = ada_grad_weight
        self.starting_lr = starting_lr
        self.lr_decay = lr_decay

    def optimize_nn(self, nn:layers, x:ndarray, y:ndarray, epochs:int, batch_size:int, loss:Loss, metric_freq:int=1, metrics=[accuracy], plt_x="epoch", plt_ys=["loss", "accuracy"]) -> DF:
        """Optimizes the neural network and returns a dataframe of the training metrics."""
        nb_batches = int(np.ceil(len(x) / batch_size))
        for epoch in range(epochs):
            if not epoch % metric_freq:
                self.record_metrics(nn, x, y, epoch, metrics, loss)
                if not plt_x is None:
                    fig = self.get_figure_widget(plt_x, plt_ys)
                    fig.data[0] = self.get_melted_training_metrics(plt_x, plt_ys)
            for batch_x, batch_y in zip(array_split(x, nb_batches), array_split(y, nb_batches)):
                self.step(nn, batch_x, batch_y, loss)
        return DF.from_records(self.training_metrics)

    @cache
    def get_figure_widget(self, plt_x, plt_ys) -> FigureWidget:
        df = self.get_melted_training_metrics(plt_x, plt_ys)
        fig = FigureWidget(scatter(df, plt_x, "value", facet_row="variable", color="varaible"))
        display(fig)
        return fig

    def get_melted_training_metrics(self, plt_x, plt_ys) -> DF:
        return DF.from_records(self.training_metrics).melt(plt_x, plt_ys)

    def record_metrics(self, nn:layers, x:ndarray, y:ndarray, epoch:int, metric_funcs:list[callable], loss:Loss) -> dict[str, any]:
        y_pred = forward(nn, x)
        self.training_metrics.append({
            "iteration": self.iterations,
            "epoch": epoch,
            "loss": loss.forward(y_pred, y),
            "learning_rate": self.learning_rate,
            **{func.__name__: func(nn=nn, y_pred=y_pred, y_true=y, loss=loss) for func in metric_funcs}
        })

    def step(self, nn:layers, x:ndarray, y:ndarray, loss:Loss):
        batch_y_preds = forward(nn, x)
        batch_gradients = loss.backward(batch_y_preds, y)
        self.update_params(batch_gradients, nn)
        self.iterations += 1

    def update_params(self, outputs_gradients:ndarray, nn:list[Layer]):
        for layer in reversed(nn):
            gradients:dict = layer.backward(outputs_gradients)
            inputs_grad = gradients.pop("inputs")
            for param_name, grads_wrt_param in gradients.items():
                param:ndarray = getattr(layer, param_name)
                gradient_wrt_param = self.postprocess_gradient(layer, param_name, grads_wrt_param)
                setattr(layer, param_name, param - gradient_wrt_param)
            outputs_gradients = inputs_grad

    def postprocess_gradient(self, layer:Layer, param_name:str, gradient_wrt_param:ndarray) -> ndarray:
        momentum = self.update_moving_average(layer, param_name + "_momentum", gradient_wrt_param, self.momentum_weight)
        cache = self.update_moving_average(layer, param_name + "_cache", gradient_wrt_param ** 2, self.ada_grad_weight)
        return self.learning_rate * momentum / (np.sqrt(cache) + EPSILON)

    def update_moving_average(self, layer, moving_average_name:str, gradient:ndarray, beta:float) -> ndarray:
        """Updates the moving average and returns it corrected."""
        moving_average = getattr(layer, moving_average_name, cached_zeros(gradient.shape)) # Get the moving averate
        moving_average = lerp(moving_average, gradient, beta) # Move the average
        setattr(layer, moving_average_name, moving_average) # Set the moved average for future use
        return moving_average / (1 - beta ** (self.iterations + 1)) # Correct and return the average 

    @property
    def learning_rate(self) -> float: 
        return self.starting_lr / (1 + self.lr_decay * self.iterations)

def lerp(a:ndarray, b:ndarray, t:float) -> ndarray:
    return a * t + (1 - t) * b

# Forward is implemented in SGD because it's the only place it's used, eventually it will be in a NueralNetwork module.
def forward(nn:list[Layer], inputs:ndarray) -> ndarray:
    return reduce(lambda x, l: l.forward(x), nn, inputs)

@cache
def cached_zeros(shape) -> ndarray:
    return np.zeros(shape)