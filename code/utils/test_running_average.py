from running_average import ExponentialMovingAverage
import matplotlib.pyplot as plt
import numpy as np
def plot_graphics(ema: ExponentialMovingAverage, num_points: int):
    values = []
    std_devs = []
    # x = list(range(num_points))
    x = lambda i : i * np.sin(i/10)
    x_values = [x(i) for i in range(num_points)]
    for i in range(num_points):
        # Обновляем EMA с новым значением (например, i)

        ema.update(x(i))
        values.append(ema.average())
        std_devs.append(ema.std_dev())

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_points), values, label='Exponential Moving Average', color='blue')
    plt.plot(range(num_points), x_values, label='x(i)', color='red')
    plt.fill_between(range(num_points), [v - sd for v, sd in zip(values, std_devs)],
                        [v + sd for v, sd in zip(values, std_devs)], color='blue', alpha=0.2, label='Standard Deviation')
    plt.title('Exponential Moving Average with Standard Deviation')
    plt.xlabel('Number of Updates')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig('ema.png')

plot_graphics(ExponentialMovingAverage(0.7), 100)