# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import torch
import torch.utils.data


class DatasetBase(torch.utils.data.Dataset):
    """
    Base class to describe a dataset.
    The dataset is made up of frames, and the frames are grouped into sequences.
    Each sequence has a name (a string).
    (A sequence could be a video, or a set of images of one scene.)
    This means they have a __getitem__ which returns an instance of a FrameData,
    which will describe one frame in one sequence.
    """

    def __len__(self) -> int:
        raise NotImplementedError()

    def sequence_names(self) -> Iterable[str]:
        """Returns an iterator over sequence names in the dataset."""
        return self._seq_to_idx.keys()

    def category_to_sequence_names(self) -> Dict[str, List[str]]:
        """
        Returns a dict mapping from each dataset category to a list of its
        sequence names.
        Returns:
            category_to_sequence_names: Dict {category_i: [..., sequence_name_j, ...]}
        """
        c2seq = defaultdict(list)
        for sequence_name in self.sequence_names():
            first_frame_idx = next(
                self.sequence_indices_in_order(sequence_name)
            )
            sequence_category = self[first_frame_idx].sequence_category
            c2seq[sequence_category].append(sequence_name)
        return dict(c2seq)

    def sequence_frames_in_order(
        self, seq_name: str, subset_filter: Optional[Sequence[str]] = None
    ) -> Iterator[Tuple[float, int, int]]:
        """Returns an iterator over the frame indices in a given sequence.
        We attempt to first sort by timestamp (if they are available),
        then by frame number.
        Args:
            seq_name: the name of the sequence.
        Returns:
            an iterator over triplets `(timestamp, frame_no, dataset_idx)`,
                where `frame_no` is the index within the sequence, and
                `dataset_idx` is the index within the dataset.
                `None` timestamps are replaced with 0s.
        """
        seq_frame_indices = self._seq_to_idx[seq_name]
        nos_timestamps = self.get_frame_numbers_and_timestamps(
            seq_frame_indices, subset_filter
        )

        yield from sorted(
            [
                (timestamp, frame_no, idx)
                for idx, (frame_no, timestamp) in zip(
                    seq_frame_indices, nos_timestamps
                )
            ]
        )

    def sequence_indices_in_order(
        self, seq_name: str, subset_filter: Optional[Sequence[str]] = None
    ) -> Iterator[int]:
        """Same as `sequence_frames_in_order` but returns the iterator over
        only dataset indices.
        """
        for _, _, idx in self.sequence_frames_in_order(seq_name, subset_filter):
            yield idx
