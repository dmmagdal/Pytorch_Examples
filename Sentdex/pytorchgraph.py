# pytorchgraph.py
# Companion program for pytorch_basics8.py
# Pytorch 1.8
# Windows/MacOS/Linux
# Python 3.7


import matplotlib.pyplot as plt
from matplotlib import style


def main():
	style.use("ggplot")
	model_name = "model-1642371666"


	def create_acc_loss_graph(model_name):
		# Load file.
		contents = open("model.log", "r").read().split("\n")

		# Metrics from log file.
		times = []
		accuracies = []
		losses = []
		val_accs = []
		val_losses = []

		# Iterate through file line contents.
		for c in contents:
			# Load metrics relevant to the specified model.
			if model_name in c:
				name, timestamp, acc, loss, val_acc, val_loss = c.split(",")
				times.append(float(timestamp))
				accuracies.append(float(acc))
				losses.append(float(loss))
				val_accs.append(float(val_acc))
				val_losses.append(float(val_loss))

		# Graph those metrics.
		fig = plt.figure()
		ax1 = plt.subplot2grid((2, 1), (0, 0))
		ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
		ax1.plot(times, accuracies, label="acc")
		ax1.plot(times, val_accs, label="val_acc")
		ax1.legend(loc=2)
		ax2.plot(times, losses, label="loss")
		ax2.plot(times, val_losses, label="val_loss")
		ax2.legend(loc=2)
		plt.show()


	create_acc_loss_graph(model_name)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()