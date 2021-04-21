#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch
import code

from torch import nn
from numpy import inf


class Trainer:
    """
    Trainer class
    """
    def __init__(self, model, train_data_loader, valid_data_loader):
        # Set up GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(1)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # Set up criterion (loss) and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.0)

        # Set up epochs and checkpoint frequency
        self.min_epochs = 5
        self.max_epochs = 100
        self.save_period = 1
        self.start_epoch = 1
        self.early_stop = 15

        # Set up data loaders for training/validation examples
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        # Set up checkpoint saving and loading
        self.checkpoint_dir = "../saved_models/"

    def train(self):
        """Complete training logic."""
        not_improved_count = 0
        best_validation_fscore = 0.0

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Perform one training epoch and output training metrics
            training_metrics = self.run_epoch(epoch, self.train_data_loader, training=True)
            print("Training epoch {} finished.".format(epoch))
            self.log_metrics(training_metrics)

            # Perform one validation epoch and output validation metrics
            validation_metrics = self.run_epoch(epoch, self.valid_data_loader, training=False)
            print("Validation epoch {} finished.".format(epoch))
            self.log_metrics(validation_metrics)

            # Check if model is new best according to validation F1 score
            improved = validation_metrics["fscore"] > best_validation_fscore
            if improved:
                best_validation_fscore = validation_metrics["fscore"]
                not_improved_count = 0
            else:
                not_improved_count += 1

            if improved or epoch % self.save_period == 0:
                self._save_checkpoint(epoch, is_best=improved)

            if not_improved_count > self.early_stop and epoch >= self.min_epochs:
                print("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                break

    def run_epoch(self, epoch, data_loader, training=False):
        """Run one epoch.

        :param epoch: Integer, current epoch number.
        :param data_loader: Data loader to fetch training examples from.
        :param training: If true, model will be trained (i.e. backpropagation happens).
        :return: A dictionary that contains information about metrics (loss, accuracy, prf)."""

        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_metrics = {"loss": 0.0}
        num_evaluated_batches = 0
        num_predictions = 0
        num_correct = 0
        tp = 0
        fp = 0
        fn = 0

        with torch.set_grad_enabled(training):
            for instance_batch in data_loader:
                # Get target labels
                target = torch.LongTensor([inst.propagation for inst in instance_batch])
                target = self._to_device(target)

                # Run model
                model_output = self.model(instance_batch)

                # Compute loss
                loss = self.criterion(model_output, target)

                # Add metrics to overall total
                epoch_metrics["loss"] += loss.item()
                num_predictions += len(target)
                model_predictions = torch.argmax(model_output, dim=1)
                num_correct += int(model_predictions.eq(target).sum())

                tp += int((model_predictions * target).sum())
                fp += int((model_predictions > target).sum())
                fn += int((model_predictions < target).sum())

                # Perform backpropagation (when training)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Print progress
                num_evaluated_batches += 1
                print('{} Epoch: {} {} Loss: {:.6f}'.format(
                    "Training" if training else "Validation",
                    epoch,
                    self._progress(num_evaluated_batches, data_loader),
                    loss.item()))

        epoch_metrics["accuracy"] = num_correct / num_predictions
        epoch_metrics["precision"] = p = tp / (tp+fp) if tp+fp else 0.0
        epoch_metrics["recall"] = r = tp / (tp+fn) if tp+fn else 0.0
        epoch_metrics["fscore"] = 2*p*r/(p+r) if p+r else 0.0

        return epoch_metrics

    def log_metrics(self, metrics):
        print("Loss: {:.2f}".format(metrics["loss"]))
        print("Accuracy: {:.2f}%".format(metrics["accuracy"] * 100))
        print("Precision: {:.2f}%".format(metrics["precision"] * 100))
        print("Recall: {:.2f}%".format(metrics["recall"] * 100))
        print("F-Score: {:.2f}%".format(metrics["fscore"] * 100))

    def _unroll_sequence_batch(self, batch):
        """Unroll a batch of sequences, i.e. flatten batch and sequence dimension. (Used for loss computation)"""
        shape = batch.shape
        if len(shape) == 3:  # Model output
            return batch.view(shape[0]*shape[1], shape[2])
        elif len(shape) == 2:  # Target labels
            return batch.view(shape[0]*shape[1])

    def _progress(self, num_completed_batches, data_loader):
        """Nicely formatted epoch progress"""
        return '[{}/{} ({:.0f}%)]'.format(num_completed_batches, len(data_loader),
                                          100.0 * num_completed_batches / len(data_loader))

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, is_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param is_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = str(self.checkpoint_dir + 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if is_best:
            best_path = str(self.checkpoint_dir + 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _load_model(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Model checkpoint loaded.")

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            assert all(isinstance(val, torch.Tensor) for val in data.values())
            data_on_device = dict()
            for key in data:
                data_on_device[key] = data[key].to(self.device)
            return data_on_device
        else:
            raise Exception("Cannot move this kind of data to a device!")
