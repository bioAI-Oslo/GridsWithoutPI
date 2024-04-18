import matplotlib.pyplot as plt
import torch 
from models import RNNGC, FFGC

m1 = FFGC()
m2 = RNNGC()

m1 = m1.load("./best_models/FFGC_markus.pkl")
m2 = m2.load("./best_models/RNNGC_markus.pkl")

fig, ax = plt.subplots(1, 1, figsize = (1.5,1.5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.plot(m1.total_loss_history, label = "Feedforward", alpha = 0.6)
plt.plot(m2.total_loss_history, "green", label = "RNN", alpha = 0.4)
plt.xlabel("Train Step")
plt.ylabel("Loss")
plt.legend(frameon = False)
plt.savefig("./figures/joint_loss_history", bbox_inches = "tight")
plt.show()