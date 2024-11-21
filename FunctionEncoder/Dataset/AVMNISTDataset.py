from abc import abstractmethod

import torch
from omegaconf import DictConfig

from FunctionEncoder.Dataset.MultiAgentDataset import MultiAgentDataset
from MultiBench.datasets.avmnist.get_data import get_dataloader


class AVMNISTDataset(MultiAgentDataset):
    """Dataset for the AVMNIST dataset."""

    def __init__(
        self,
        logit_scale: float = 5,
        split: str = "train",
        heldout_classes: list[int] | None = None,
        heldout_classes_only: bool = False,
        n_functions_per_sample: int = 10,
        n_examples_per_sample: int = 100,
        n_points_per_sample: int = 100,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
        cfg: DictConfig | None = None,
    ) -> None:
        """Initialize the AVMNIST dataset.

        Args:
            logit_scale (float): The logit scale for the dataset.
            split (str): The split to use for the dataset.
            heldout_classes (list): The classes to use for heldout.
            heldout_classes_only (bool): Whether to only use heldout classes.
            n_functions_per_sample (int): The number of functions to sample per sample.
            n_examples_per_sample (int): The number of examples to sample per function.
            n_points_per_sample (int): The number of points to sample per function.
            device (str): The device to use for the dataset.
            dtype (torch.dtype): The dtype to use for the dataset.
        """
        if heldout_classes is None:
            heldout_classes = [9]

        super().__init__(
            input_size=(3, 32, 32),
            output_size=(2,),
            total_n_functions=100,
            total_n_samples_per_function=100,
            data_type="categorical",
            n_functions_per_sample=n_functions_per_sample,
            n_examples_per_sample=n_examples_per_sample,
            n_points_per_sample=n_points_per_sample,
            device=device,
            dtype=dtype,
        )
        assert split.lower() in [
            "train",
            "valid",
            "test",
        ], "split must be 'train', 'valid', or 'test'"
        # torch dataloader
        trains, valids, tests = get_dataloader(
            cfg.avmnist.dataset_path, train_shuffle=False, batch_size=64
        )
        if split.lower() == "train":
            self.dataloader = trains
        elif split.lower() == "valid":
            self.dataloader = valids
        else:
            self.dataloader = tests

        self.classes = list(range(10))
        self.data_tensor = []
        for batch in self.dataloader:
            images, audio, labels = batch  # B x (1, 28, 28), B x (1, 112, 112), B

        # figure out which indicies we can sample from
        self.heldout_indicies = torch.tensor(
            [self.classes.index(class_name) for class_name in heldout_classes],
            device=self.device,
            dtype=torch.int64,
        )
        self.training_indicies = torch.tensor(
            [i for i in range(len(self.classes)) if i not in self.heldout_indicies],
            device=self.device,
            dtype=torch.int64,
        )

        # logit scale is used to specify how sharp the emprical distribution is. Higher values make the distribution sharper.
        self.logit_scale = logit_scale
        self.heldout_classes_only = heldout_classes_only

    @abstractmethod
    def sample(
        self, heldout=False
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, dict]:
        with torch.no_grad():
            # first randomly sample which classes to train on
            classes = self.sample_classes(heldout or self.heldout_classes_only)

            # next, sample positive examples, ie images that belong to the class
            positive_example_xs, positive_example_class_indicies = (
                self.sample_positive_examples(classes, self.n_examples_per_sample // 2)
            )
            positive_xs, positive_class_indicies = self.sample_positive_examples(
                classes, self.n_points_per_sample // 2
            )

            # next, sample negative examples, ie random images from other classes
            negative_example_xs, negative_example_class_indicies = (
                self.sample_negative_examples(classes, self.n_examples_per_sample // 2)
            )
            negative_xs, negative_class_indicies = self.sample_negative_examples(
                classes, self.n_points_per_sample // 2
            )

            # concatenate the positive and negative examples
            example_xs = torch.cat([positive_example_xs, negative_example_xs], dim=1)
            xs = torch.cat([positive_xs, negative_xs], dim=1)

            # generate the ground truth labels
            example_ys = self.logit_scale * torch.ones(
                (self.n_functions_per_sample, self.n_examples_per_sample, 2),
                device=self.device,
            )
            ys = self.logit_scale * torch.ones(
                (self.n_functions_per_sample, self.n_points_per_sample, 2),
                device=self.device,
            )
            example_ys[:, : self.n_examples_per_sample // 2, 1] *= -1
            example_ys[:, self.n_examples_per_sample // 2 :, 0] *= -1
            ys[:, : self.n_points_per_sample // 2, 1] *= -1
            ys[:, self.n_points_per_sample // 2 :, 0] *= -1

            # fetch relevant info for plotting
            info = {
                "classes_idx": classes,
                "class_labels": [self.classes[class_idx] for class_idx in classes],
                "positive_example_class_indicies": positive_example_class_indicies,
                "positive_class_indicies": positive_class_indicies,
                "negative_example_class_indicies": negative_example_class_indicies,
                "negative_class_indicies": negative_class_indicies,
            }

            # return the data
            return example_xs, example_ys, xs, ys, info

    # samples which classes we will use for this batch
    def sample_classes(self, heldout):
        if heldout:
            if self.n_functions_per_sample >= len(self.heldout_indicies):
                classes = self.heldout_indicies
            else:
                perm = torch.randperm(len(self.heldout_indicies), device=self.device)[
                    : self.n_functions_per_sample
                ]
                classes = self.heldout_indicies[perm]
        else:
            perm = torch.randperm(len(self.training_indicies), device=self.device)[
                : self.n_functions_per_sample
            ]
            classes = self.training_indicies[perm]
        return classes

    # samples images randomly from the classes given
    def sample_positive_examples(self, classes, count):
        example_indicies = torch.stack(
            [
                torch.randperm(len(self.data_tensor[1]), device=self.device)[:count]
                for class_idx in classes
            ]
        )
        example_indicies = example_indicies.reshape(
            *example_indicies.shape, 1, 1, 1
        ).expand(
            -1,
            -1,
            self.data_tensor.shape[-3],
            self.data_tensor.shape[-2],
            self.data_tensor.shape[-1],
        )
        examples = self.data_tensor[classes].gather(dim=1, index=example_indicies)
        return examples, classes

    # samples images randomly that ARENT from the classes given
    def sample_negative_examples(self, classes, count):
        # get random indicies of classes
        class_indicies = torch.zeros(
            (len(classes), count), device=self.device, dtype=torch.int64
        )
        for i in range(len(classes)):
            acceptable_indicies = self.training_indicies[
                self.training_indicies != classes[i]
            ]
            perm = torch.randint(
                0, len(acceptable_indicies), (count,), device=self.device
            )
            class_indicies[i] = acceptable_indicies[perm]

        # get random indicies of images
        example_indicies = torch.stack(
            [
                torch.randperm(len(self.data_tensor[1]), device=self.device)[:count]
                for class_idx in classes
            ]
        )

        # get the images
        examples = self.data_tensor[
            class_indicies.reshape(-1), example_indicies.reshape(-1)
        ]
        examples = examples.reshape(
            len(classes),
            count,
            examples.shape[-3],
            examples.shape[-2],
            examples.shape[-1],
        )
        return examples, class_indicies
