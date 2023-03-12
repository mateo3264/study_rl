import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(2, 1, 1)


def update(i):
    graph_data = open('terminal_rewards.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = np.arange(len(lines)-1)
    ys_pos = []
    ys_neg = []
    epss = []
    for line in lines:
        if len(line) > 1:

            y_pos = int(line.split(',')[0])
            epsilon = float(line.split(',')[2])
            
            if y_pos == 1:
                ys_pos.append(y_pos)
                ys_neg.append(0)
            else:
                ys_pos.append(0)
                ys_neg.append(1)
            epss.append(epsilon)

    ax1.clear()
    ax1.step(xs, np.cumsum(ys_pos), label='r+')
    ax1.step(xs, np.cumsum(ys_neg), label='r-')
    #ax2.plot(xs, epss)
    ax1.legend()
    ax1.set_title('Positive Terminal Reward vs Negative Terminal Reward')
    
    


def show_dyn_graph():
    ani = animation.FuncAnimation(fig, update, interval=1000)
    plt.show()

if __name__ == '__main__':
    show_dyn_graph()
