
# CPU & GPU Approach:

The typical computation process is sequential i.e. task are assigned in queues and it is surved according to its arrival order LIFO. The normal core CPU can compute its fastly but the limitation of this approach is that it serves other process after completing the ones it sees first and CPU has limited numbers of threads. On the other hand GPU has handrads/thousands of thread that can compute parrallely. The following workflow will give more insight:

**CPU approach:**

<img src='https://github.com/hamzafar/GPU_Computation/blob/master/images/sequential%20programming.PNG?raw=true'>


**GPU approach:**

<img src='https://github.com/hamzafar/GPU_Computation/blob/master/images/GPU%20procesing.PNG?raw=true'>



# Compare performance 

To validate above we have compared the performance between sequential programming (CPU) and parallel programming (GPU) in terms of time.

Random data is generated and then simply multiplied using simple approach (for loop) and same data is multiplied on GPU in tensorflow multipy function.

The results shows that with simple arithmetic operations GPU outperformed.

### Random Data


```python
import numpy as np
import time

import tensorflow as tf
```


```python
# Number of rows to be multiplied
N = 50000
```


```python
# Random data genrated for multiplication 
#with different seed

np.random.seed(10)
df = np.random.randn(N)

np.random.seed(100)
fms = np.random.randn(N)

np.random.seed(1000)
gff = np.random.randn(N)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
np.random.seed(10000)
cof = np.random.randn(N)

pof = np.ndarray(N)
risk = np.ndarray(N)
```

### Sequential Programming


```python
# determine the start time
start_time = time.time()

# multiply each of the variable
for i in range(0, len(cof)):
    pof[i] = df[i] * fms[i] * gff[i]

for i in range(0, len(pof)):
    risk[i] = pof[i] * cof[i]
# find out the runtime
print("--- %s seconds ---" % (time.time() - start_time))
```

    --- 0.09100532531738281 seconds ---
    

## Parallel Programing


```python
# place data to tensors
t_df = tf.constant(df, name= 'df')
t_fms = tf.constant(fms, name= 'fms')
t_gff = tf.constant(gff, name= 'gff')
t_cof = tf.constant(cof, name= 'cof')
```


```python
# multiply tensors
t_pof = tf.multiply(tf.multiply(t_df, t_fms), t_gff , name='pof')
t_risk = tf.multiply(t_pof, t_cof, name='cof')
```


```python
# start tensorflow session
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
```


```python
# determine the start time
start_time = time.time()

sess.run(t_risk)
# find out the runtime
print("--- %s seconds ---" % (time.time() - start_time))
```

    --- 0.21601223945617676 seconds ---
    


```python
# sess.close()
```

# Proposed Solution:

On board appraoch we will read data from the Data Base to GPU in vector/Matrix form. These Vectors have n number of rows; each row in the tensor(vector/matrix) is assign to specific thread in GPU. 

There might be scanrios where we need to compute different subset of data, so we will break the data according to given condition and these subsets will be computed parrallelrly. The approach is shown below:

<img src= 'https://github.com/hamzafar/GPU_Computation/blob/master/images/tensor.PNG?raw=true'>

# Computation Graph

The cool thing about this approach, visualization graph in which we can validate computation algorithm. As we can visulaize the the input fileds like dff, pof; the steps where they are get multiplied. All process flow is handy. 

<img src = "https://github.com/hamzafar/GPU_Computation/blob/master/images/Computation%20graph.PNG?raw=true">

In the computation, we multiplied three variable gff, fms and df that results in pof which then multiplied by cof to yield in risk. we can easily observe the process in the above graph


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import numpy as np

from IPython.display import clear_output, Image, display, HTML
```


```python
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script src="//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js"></script>
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
```


```python
# show_graph(sess.graph)
```


