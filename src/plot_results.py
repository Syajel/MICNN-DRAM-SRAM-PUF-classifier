import matplotlib.pyplot as plt


def plot(e,arr1,arr2,label1,label2,plotname,x,y,t):
    """
    Plots accuracy or loss graphs for both SRAM and DRAM
    e: number of epochs
    arr1: array of datapoints of the first line
    arr2: array of datapoints of the second line
    label1: label of the first line
    label2: label of the second line
    plotname: file name
    x: x-axis label
    y: y-axis label
    t: figure title
    """
    epochs = list(range(1,e+1))
    # Plot the datapoints
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, arr1, marker='o', linestyle='-', color='b', label=label1)
    plt.plot(epochs, arr2, marker='s', linestyle='--', color='r', label=label2)

    # Labels and Title
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(t)
    plt.legend()
    plt.grid(True)

    # Save the plot as PNG
    plot_path = f"plots/{plotname}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')



