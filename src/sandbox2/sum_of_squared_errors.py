import numpy as np
activity = np.array([1, 2, 3, 4, 5])
sum_squared_activities = np.dot(activity.T, activity)

print(sum_squared_activities)

sum_squared_activities = np.sum(np.square(activity))

print(sum_squared_activities)
