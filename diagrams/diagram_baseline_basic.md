```mermaid
graph TD
    %% -------------------
    %% 1. Data Loading Phase
    %% -------------------
    subgraph Data_Pipeline [Step 1: Data Loading & Preprocessing]
        Files[("PTB-XL Data<br/>(Waveforms + CSV)")] -->|load_raw_data| Load[Load Raw Signals]
        Load -->|Filter| Filter[Select NORM vs MI]
        Filter -->|Z-Score| Norm["Normalization<br/>(Mean=0, Std=1 per lead)"]
        Norm --> Batch[("Input Batch X<br/>Shape: [Batch, 12, L]")]
    end

    %% -------------------
    %% 2. The Model Architecture
    %% -------------------
    subgraph IMN_Model [Step 2: Interpretable Mesomorphic Network]
        
        %% Path A: The Hypernetwork (Generates Weights)
        Batch -->|"Reshape [B,1,12,L]"| CNN_In[CNN Input]
        CNN_In -->|Conv2D Layers| Conv1[Conv Block 1]
        Conv1 -->|Conv2D + MaxPool| Conv2[Conv Block 2]
        Conv2 -->|Conv2D + MaxPool| Conv3[Conv Block 3]
        Conv3 -->|Global Avg Pool| Feature[("Latent Features<br/>Shape: [Batch, 64]")]
        Feature -->|Linear Projection| HyperHead[HyperHead Layer]
        HyperHead -->|Reshape| Params{Generated Parameters}
        
        Params -->|Split| W_gen[("Generated Weights (W)<br/>Shape: [Batch, Classes, 12*L]")]
        Params -->|Split| b_gen[("Generated Bias (b)<br/>Shape: [Batch, Classes, 1]")]

        %% Path B: The Input Application
        Batch -->|Flatten| X_flat[("Flattened Input X<br/>Shape: [Batch, 12*L]")]
        
        %% Convergence
        W_gen --> DotProd("Dot Product")
        X_flat --> DotProd
        DotProd -->|Sum over features| WeightedSum["Weighted Sum"]
        WeightedSum -->|Add Bias| Logits["Final Logits<br/>Shape: [Batch, Classes]"]
        b_gen --> Logits
    end

    %% -------------------
    %% 3. Training & Loss
    %% -------------------
    subgraph Optimization [Step 3: Loss & Training]
        Logits -->|Softmax| Pred[Prediction]
        Labels[True Labels y] --> CE
        Pred --> CE[CrossEntropy Loss]
        
        W_gen -->|Abs Mean| L1["L1 Regularization<br/>(Enforces Sparsity)"]
        
        CE --> TotalLoss["Total Loss = CE + λ * L1"]
        L1 --> TotalLoss
        TotalLoss -->|Backprop| Update[Update CNN Weights]
    end

    %% -------------------
    %% 4. XAI (Explainability)
    %% -------------------
    subgraph XAI ["Step 4: Visualization - Inference Only"]
        W_gen -->|Select Class 1| W_pos[Weights for MI]
        Batch -->|Original X| X_vis
        
        W_pos -.->|Element-wise Multiply| Impact["Feature Impact<br/>(Impact = W * X)"]
        X_vis -.-> Impact
        Impact -->|Segment Aggregation| Heatmap[Generate Heatmap]
        Heatmap --> PDF[Save PDF Report]
    end