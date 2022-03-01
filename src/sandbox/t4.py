import numpy as np
from matplotlib import pyplot as plt

# Objective function
def f(x,extra=[]):
    return x**2 - 2*x + 3

# Function to compute the gradient
def grad(x,extra=[]):
    return 2*x - 2

def visualize_fw():
    xcoord = np.linspace(-10.0,10.0,50)
    ycoord = np.linspace(-10.0,10.0,50)
    w1,w2 = np.meshgrid(xcoord,ycoord)
    pts = np.vstack((w1.flatten(),w2.flatten()))
    
    # All 2D points on the grid
    pts = pts.transpose()
    
    # Function value at each point
    f_vals = np.sum(pts*pts,axis=1)
    function_plot(pts,f_vals)
    plt.title('Objective Function Shown in Color')
    plt.show()
    return pts,f_vals

def annotate_pt(text,xy,xytext,color):
    plt.plot(xy[0],xy[1],marker='P',markersize=10,c=color)
    plt.annotate(text,xy=xy,xytext=xytext,
                 # color=color,
                 arrowprops=dict(arrowstyle="->",
                 color = color,
                 connectionstyle='arc3'))

def visualize_learning(w_history):  
    
    # Make the function plot
    function_plot(pts, f_vals)
    
    # Plot the history
    plt.plot(w_history[:,0],w_history[:,1],marker='o',c='magenta') 
    
    # Annotate the point found at last iteration
    annotate_pt('minimum found',
                (w_history[-1,0],w_history[-1,1]),
                (-1,7),'green')
    iter = w_history.shape[0]
    for w,i in zip(w_history,range(iter-1)):
        # Annotate with arrows to show history
        plt.annotate("",
                    xy=w, xycoords='data',
                    xytext=w_history[i+1,:], textcoords='data',
                    arrowprops=dict(arrowstyle='<-',
                            connectionstyle='angle3'))     

def function_plot(pts,f_val):
    f_plot = plt.scatter(pts[:,0],pts[:,1],
                         c=f_val,vmin=min(f_val),vmax=max(f_val),
                         cmap='RdBu_r')
    plt.colorbar(f_plot)
    # Show the optimal point
    annotate_pt('global minimum',(0,0),(-5,-7),'yellow')

pts,f_vals = visualize_fw() 

def gradient_descent(max_iterations,threshold,w_init, obj_func,grad_func,extra_param = [], learning_rate=0.05, momentum=0.8):
    
    w = w_init
    w_history = w
    f_history = obj_func(w,extra_param)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    
    while  i<max_iterations :
        delta_w = -learning_rate*grad_func(w,extra_param) + momentum*delta_w
        w = w + delta_w
        
        # store the history of w and f
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,obj_func(w,extra_param)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
    
    return w_history,f_history

#def f(x):
#        return (x**3)

#x = np.linspace(-2,3,100)


#y = (x - 1)**2 + 2


#y = (x**4/4) - (x**3/3) - x**2 + 2

#y = x**3

#y =  x**2 - 2*x + 3
rand = np.random.RandomState(19)
w_init = rand.uniform(-10,10,2)

def solve_fw():
        ind = 1
        plt.subplot(3,4,ind)        
        w_history,f_history = gradient_descent(5, -1, w_init, f,grad,[], 0.05)
        visualize_learning(w_history)
        plt.show()
        ind = ind+1

solve_fw()

#plt.plot(x, y)
#plt.show()