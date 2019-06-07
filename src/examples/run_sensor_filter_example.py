import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series
import src.problemgenerator.copy as copy

x = np.arange(100.0)
y = np.arange(100.0, 200.0)
print(x)
print(y)
data = (x, y)

x_node = array.Array(x[0].shape)
y_node = array.Array(y[0].shape)
series_node = series.TupleSeries([x_node, y_node])
root_node = copy.Copy(series_node)

# Every increase in time results in drift increasing by 0.1
# y_node.addfilter(filters.SensorDrift(magnitude=0.1))

def strange(a):
    if a <= 170 and a >= 150:
        return 1729

    return a

# y_node.addfilter(filters.StrangeBehaviour(strange))
y_node.addfilter(filters.Gap(max_length=5, grace_period=5))
output = root_node.process(data)

print("x output", output[0])
print("y output", output[1])
print("x output", output[0].dtype)
print("y output", output[1].dtype)
