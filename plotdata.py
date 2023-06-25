import matplotlib.pyplot as plt
import numpy as np


def get_costs(path, n):
    with open(path, "r") as f:
        lines = f.readlines()
        costs = []
        for l in lines:
            if "Cost:" in l:
                if "tensor" in l:
                    cost = float(l.replace("Cost: tensor(", "").replace(")", ""))
                else:
                    cost = 0.0
                costs.append(cost)
        subs = np.array([costs[i::n] for i in range(n)]).mean(axis=1)
        return subs


def get_numbered_y(lines, n):
    i = 0
    outer_a = []
    inner_a = []
    for l in range(len(lines)):
        numbers_str = lines[l].split(" ")
        numbers = list(map(lambda n: float(n), numbers_str))
        inner_a.append(numbers)
        i += 1
        if i > n-1:
            i = 0
            outer_a.append(np.array([min(item) for item in inner_a]))
            inner_a = []

    return outer_a

def get_ys(path, n):
    with open(path, "r") as f:
        lines = f.readlines()
        cleaned_lines = []
        for l in lines:
            if "Y:" in l:
                cleaned_l = l.replace("\n", "").replace("array(", "").replace(")", "")\
                    .replace("[", "").replace("]", "").replace("Y: ", "").replace(" dtype=float32", "").replace(",", "")
                cleaned_lines.append(cleaned_l)
        ys = get_numbered_y(cleaned_lines,n)
        return ys

def get_data_new(path, n):
    with open(path, "r") as f:
        lines = f.readlines()
        cleaned_ylines = []
        cleaned_clines = []
        c_lines = []
        y_lines = []
        for i in range(len(lines)):
            if i % 2:
                c_lines.append(lines[i][4:])
            else:
                y_lines.append(lines[i][4:])
        for l in y_lines:
            cleaned_l = l.replace("\n", "").replace("array(", "").replace(")", "") \
                .replace("[", "").replace("]", "").replace("Y: ", "").replace(" dtype=float32", "").replace(",", "")
            cleaned_ylines.append(cleaned_l)
        ys = get_numbered_y(cleaned_ylines, n)
        costs = []
        for c in c_lines:
            if "tensor" in c:
                cost = float(c.replace("tensor(", "").replace(")", ""))
            else:
                cost = 0.0
            costs.append(cost)
        costs = np.array([costs[i::n] for i in range(n)]).mean(axis=1)

        return ys, costs


c_ei = get_costs("results/ei.txt", 5)
y_ei = get_ys("results/ei.txt", 5)
mc_ei = (np.array(c_ei) * 10).tolist()
my_ei = (np.array(y_ei).mean(axis=0) * 100).tolist()
z_ei = [x for _, x in sorted(zip(mc_ei, my_ei))]
mc_ei = sorted(mc_ei)

c_kg = get_costs("results/kg.txt", 5)
y_kg = get_ys("results/kg.txt", 5)
mc_kg = (np.array(c_kg) * 10).tolist()
my_kg = (np.array(y_kg).mean(axis=0) * 100).tolist()
z_kg = [x for _, x in sorted(zip(mc_kg, my_kg))]
mc_kg = sorted(mc_kg)

c_takg = get_costs("results/takg.txt", 5)
y_takg = get_ys("results/takg.txt", 5)
mc_takg = (np.array(c_takg) * 10).tolist()
my_takg = (np.array(y_takg).mean(axis=0) * 100).tolist()
z_takg = [x for _, x in sorted(zip(mc_takg, my_takg))]
mc_takg = sorted(mc_takg)


with plt.style.context("seaborn-v0_8-dark"):
    plt.rc('font', size=20)
    plt.title("Feedforward neural network on MNIST")
    plt.xlabel("Cost (resource used)")
    plt.ylabel("Validation Error (%)")
    plt.plot(mc_ei, z_ei, marker='x', label='EI', linewidth=3, markersize=8)
    plt.plot(mc_kg, z_kg, marker='x', label='KG', linewidth=3, markersize=8)
    plt.plot(mc_takg, z_takg, marker='x', label='taKG', linewidth=3, markersize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results1.png', dpi=220)

    plt.show()