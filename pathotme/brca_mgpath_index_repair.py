"""Narrow BRCA pandas column-index repair, shared by PLIP and RN50 runners."""
import numpy as np
import torch
from pathotme.guided_mgpath_adapter import one
from pathotme.locked_models import BreastGuidedMGPathMethod


class IndexedBreastGuidedMGPathMethod(BreastGuidedMGPathMethod):
    def inputs(self,batch):
        low,low_edges,high,high_edges,label=self._unpack(batch,self.device)
        metadata=batch[-2]
        # A tuple of column names is interpreted as a multidimensional .loc
        # key for a scalar row. An explicit list selects the ordered panel.
        values=self.tme.loc[one(metadata,'slide_id'),list(self.names)].to_numpy(dtype=np.float32)
        tme=torch.as_tensor(values,device=self.device).unsqueeze(0)
        return (low,low_edges,high,high_edges,tme),label,metadata
