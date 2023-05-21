import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def read_file(filename):
	out = []
	with open(filename, "r") as file:
		for row in file:
			out.append(row.strip().split(","))
	
	return out

img_name = sys.argv[1]

horHomoCues = [float(num) for num in read_file(f"out/{img_name}_horHomogeneity.csv")[0]]
verHomoCues = [float(num) for num in read_file(f"out/{img_name}_verHomogeneity.csv")[0]]
x = np.arange(0, len(horHomoCues))
x2 = np.arange(0, len(verHomoCues))

plt.figure(f"{img_name} homogeneity", dpi = 120)
plt.title(f"Homogeneity with variable displacement ({img_name})")
plt.xlabel("Displacement/Offset")
plt.ylabel("Homogeneity")
plt.plot(x2, verHomoCues, ".-", color = "red", label = "Vertical")
plt.plot(x, horHomoCues, ".-", color = "blue", label = "Horizontal")
plt.grid(color = "darkgray", linestyle = "dashed", linewidth = "0.5")
plt.legend()

try:
	arg = sys.argv[2]
	match arg:
		case "--save":
			if(not os.path.isdir("./out/img/")):
				os.mkdir("./out/img/")
				plt.savefig(f"./out/img/{img_name}_homogeneity.png", dpi = 280)
			else:
				plt.savefig(f"./out/img/{img_name}_homogeneity.png", dpi = 280)

		case "--show":
			plt.show()

		case "--save-and-show":
			if(not os.path.isdir("./out/img/")):
				os.mkdir("./out/img/")
				plt.savefig(f"./out/img/{img_name}_homogeneity.png", dpi = 280)
				plt.show()
			else:
				plt.savefig(f"./out/img/{img_name}_homogeneity.png", dpi = 280)
				plt.show()

		case _:
			print(
	    """Error while processing arguments. You can:
\t--save: Save figure in out/img/.
\t--show: Opens a window showing the figure.
\t--save-and-show: Save figure in out/img/ and open a window showing the figure.
			""")
except:
  pass
