# **Data Parallelism using Horovod and PyTorch Distributed**

This guide covers two popular approaches for implementing multi-node distributed training using PyTorch: **DistributedDataParallel (DDP)** and **Horovod**.

---

## **1. PyTorch DistributedDataParallel (DDP)**

DDP is a built-in feature of PyTorch that enables data parallelism across multiple nodes. It is the most widely used technique for parallelism in PyTorch.

### **Key Concepts in DDP**

- **Data Parallelism**:  
  Divides the dataset into multiple batches, each distributed across different GPUs or nodes. This allows parallel processing of data, improving throughput. Data distribution can be done via:  
  - **Weak scaling**: Increasing data size per node.  
  - **Strong scaling**: Keeping data size constant while increasing computational resources.

- **Gradient Aggregation**:  
  Computes gradients for each batch independently, then averages these gradients across GPUs/nodes. This ensures all data contributes to the final model updates while maintaining parallelism.

- **Rank**:  
  Assigns a unique identifier to each process, which helps in process coordination and identification.

### **How DDP Works**

1. Training data is divided across GPUs.  
2. The model is replicated and distributed to each GPU.  
3. Gradients are calculated on each shard of data in parallel.  
4. Gradients are aggregated (averaged) across GPUs using a reduction function.  
5. The optimizer updates the model weights.  
6. Steps 3–5 are repeated until the model converges or the specified number of epochs is completed.

### **DDP Diagram**
![DDP Diagram](https://github.com/user-attachments/assets/735f28fd-8309-4a1a-a234-cbc7914414bb)

---

## **2. Horovod**

Horovod is an alternative tool for implementing data parallelism, offering support for additional communication protocols beyond what DDP provides.

### **Key Concepts in Horovod**

- **AllReduce**:  
  Aggregates and averages gradients across all processes. Commonly used for distributed training.  

- **Broadcast**:  
  Sends initial model weights from one process to all other processes in the group, ensuring consistency at the start of training.

- **Distributed Optimizer**:  
  Wraps the model’s optimizer function for distributed use. The gradients are aggregated, and the optimizer updates weights collectively.

### **How Horovod Works**

1. The dataset is divided into mini-batches.  
2. Each mini-batch is assigned to a GPU.  
3. Initial model weights are broadcasted to all GPUs.  
4. Gradients are computed locally on each GPU.  
5. Gradients are synchronized using the AllReduce operation.  
6. Each GPU updates its local weights using a distributed optimizer.  
7. Steps 4–6 are repeated until training completes (either when epochs are finished or the model converges).  
8. Final weights are extracted as model outputs.

### **Horovod Diagram**
![Horovod Diagram](https://github.com/user-attachments/assets/c93ee090-ae3e-466e-938d-86c79da26529)

### **Explanation**
1. ML/DL applications are first loaded into PyTorch.  
2. Horovod integrates with PyTorch to distribute data and manage training via protocols like **MVAPICH**.  

---

### **Comparison: DDP vs. Horovod**

| Feature               | DDP                                   | Horovod                               |
|-----------------------|---------------------------------------|---------------------------------------|
| **Library Support**   | Built-in PyTorch                     | External library                      |
| **Communication**     | Gradient reduction (Reduce, AllReduce)| AllReduce, Broadcast                  |
| **Optimizer**         | Handled within PyTorch               | Distributed Optimizer support         |
| **Protocols**         | Supports common PyTorch protocols    | Broader protocol support (e.g., MVAPICH)|

---

## **Results**

### **1. Weak Scaling**
- **DDP Performance**  
  ![Weak Scaling with DDP](https://github.com/user-attachments/assets/42352b56-c4c3-4454-8d8a-14c3e78288ae)  
  - Observations: As expected, throughput increases with the number of nodes, but the scaling efficiency decreases slightly due to communication overhead.

- **Horovod Performance**  
  ![Weak Scaling with Horovod](https://github.com/user-attachments/assets/3ea5d304-d8ce-45ee-aa46-7d8febeebdbd)  
  - Observations: Similar to DDP, throughput improves with additional nodes, although the performance is marginally lower compared to DDP due to differences in communication protocols.

---

### **2. Strong Scaling**
- **DDP Performance**  
  ![Strong Scaling with DDP](https://github.com/user-attachments/assets/925b231c-8823-4320-b55c-a7cc44ba8bd8)  
  - Observations: While task completion time decreases with more nodes, the speedup diminishes as batch size per GPU decreases, leading to increased communication overhead.

- **Horovod Performance**  
  ![Strong Scaling with Horovod](https://github.com/user-attachments/assets/838a10fc-96cc-43fe-a7a4-df8428e31895)  
  - Observations: Similar trends are observed as with DDP. However, the performance lags slightly behind DDP due to its reliance on all-reduce communication, which is less efficient in some scenarios.

---
# **Conclusion**

This project explores techniques to optimize batch size and leverage distributed training using **PyTorch DistributedDataParallel (DDP)** and **Horovod**. The goal is to maximize training efficiency by analyzing throughput and scaling performance.

---

## **Finding the Optimal Batch Size**

To determine the optimal batch size, we tested values ranging from 64 to 1024. The batch size with the highest throughput was selected, as higher throughput reflects better performance and resource utilization.

---

## **Parallelism Techniques**

After identifying the optimal batch size, we implemented distributed training using **DDP** and **Horovod**, evaluating two scaling strategies:

### **1. Weak Scaling**
- **Concept**: The batch size per GPU is kept constant, aiming to maintain execution time while maximizing the amount of work performed in parallel.  
- **Expected Behavior**: Ideally, the speedup should scale with the number of nodes. However, some computational resources are used for inter-process communication, resulting in sub-linear scaling.  

### **2. Strong Scaling**
- **Concept**: The total batch size remains constant while the number of nodes increases. This reduces the batch size per GPU, focusing on minimizing task completion time.  
- **Observations**:  
  - As the number of nodes increases, the computation per node decreases.  
  - The proportion of resources spent on communication increases, leading to diminishing returns in speedup.

---

## **Performance Comparison: DDP vs. Horovod**

Both **DDP** and **Horovod** were evaluated for throughput and scalability. While their performance was similar, **DDP** showed a slight advantage (under 10%). This can be attributed to two factors:

1. **Integration with PyTorch**:  
   DDP, being a native PyTorch solution, utilizes PyTorch’s optimization capabilities more effectively than Horovod, which is a standalone framework.

2. **Communication Strategies**:  
   - **DDP** uses an **all-to-all** communication protocol, which generally performs better in distributed environments.  
   - **Horovod**, on the other hand, relies on **all-reduce**, which may result in slightly higher communication overhead.

---

## **Key Takeaways**

- Weak scaling is ideal for maximizing throughput while maintaining a constant time per epoch.
- Strong scaling can minimize execution time but becomes less efficient as the number of nodes increases.
- **DDP** is slightly more efficient than **Horovod** for distributed training in PyTorch, making it a preferable choice for scenarios where PyTorch is already in use.

---

This project highlights the importance of selecting the right parallelism approach and tools to maximize efficiency in distributed training workflows.


