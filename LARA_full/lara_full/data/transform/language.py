from typing import Sequence

from pydantic import Field
from transformers import PreTrainedTokenizerBase
from lara_full.data.transform.base import ModalityTransform


class ConcatLanguage(ModalityTransform):
    def apply(self, data: dict) -> dict:
        datas = []
        for key in self.apply_to:
            if isinstance(data[key], str):
                datas.append(data[key])
            else:
                datas.append(" ".join(data[key]))

        language = " ".join(datas)
        data["language"] = language

        return data
