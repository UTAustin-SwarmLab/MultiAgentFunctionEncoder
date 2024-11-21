"""Example of using the MultiAgentFunctionEncoder on the AVMNIST dataset."""

from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig

from FunctionEncoder import (
    DistanceCallback,
    FunctionEncoder,
    ListCallback,
    TensorboardCallback,
)
from FunctionEncoder.Dataset.AVMNISTDataset import AVMNISTDataset


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main function for the AVMNIST example.

    Args:
        cfg (DictConfig): The configuration for the experiment.
    """
    # hyper params
    epochs = cfg.avmnist.epochs
    n_basis = cfg.avmnist.n_basis
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_method = cfg.avmnist.train_method
    seed = cfg.seed
    load_path = cfg.avmnist.load_path
    residuals = cfg.avmnist.residuals
    if load_path is None or load_path == "":
        logdir = f"logs/avmnist_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"  # noqa: DTZ005
    else:
        logdir = load_path

    # seed torch
    torch.manual_seed(seed)

    # create a dataset
    dataset = AVMNISTDataset(cfg=cfg)

    if load_path is None:
        # create the model
        model = FunctionEncoder(
            input_size=dataset.input_size,
            output_size=dataset.output_size,
            data_type=dataset.data_type,
            n_basis=n_basis,
            method=train_method,
            model_type="CNN",
            use_residuals_method=residuals,
        ).to(device)

        # create callbacks
        cb1 = TensorboardCallback(logdir)  # this one logs training data
        cb2 = DistanceCallback(
            dataset, tensorboard=cb1.tensorboard
        )  # this one tests and logs the results
        callback = ListCallback([cb1, cb2])

        # train the model
        model.train_model(dataset, epochs=epochs, callback=callback)

        # save the model
        torch.save(model.state_dict(), f"{logdir}/model.pth")
    else:
        # load the model
        model = FunctionEncoder(
            input_size=dataset.input_size,
            output_size=dataset.output_size,
            data_type=dataset.data_type,
            n_basis=n_basis,
            method=train_method,
            use_residuals_method=residuals,
        ).to(device)
        model.load_state_dict(torch.load(f"{logdir}/model.pth"))

    # get a new dataset for testing
    # ID test
    example_xs, example_ys, xs, ys, info = dataset.sample()
    y_hats = model.predict_from_examples(example_xs, example_ys, xs)

    # OOD Test
    example_xs, example_ys, xs, ys, info = dataset.sample(
        heldout=True
    )  # heldout classes
    y_hats = model.predict_from_examples(example_xs, example_ys, xs)


if __name__ == "__main__":
    main()
