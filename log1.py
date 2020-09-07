"""from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

ea = event_accumulator.EventAccumulator('./runs/image/events.out.tfevents.1598385359.Sanas-MacBook-Pro.local.98714.0',
size_guidance={event_accumulator.COMPRESSED_HISTOGRAMS: 500, event_accumulator.IMAGES: 4, event_accumulator.AUDIO: 4, event_accumulator.SCALARS: 0,
 event_accumulator.HISTOGRAMS: 1,})

ea.Reload() # loads events from file
ea.Tags()
ea.Scalars('Loss')
pd.DataFrame(ea.Scalars('Loss')).to_csv('Loss.csv')"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)