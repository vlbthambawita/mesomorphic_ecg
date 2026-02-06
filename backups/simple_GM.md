```mermaid
flowchart TD
    %% Define styles
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef layer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef tensor fill:#fff9c4,stroke:#fbc02d,stroke-width:1px,stroke-dasharray:5\,5;
    classDef param fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px;

    %% Input
    Input[("Input Signal x<br>(B, 12, L)")]:::input
    
    %% Expand dim
    ExpInput["Unsqueeze(1)<br>(B, 1, 12, L)"]:::tensor
    Input --> ExpInput

    %% ENCODER / BACKBONE
    subgraph Encoder ["Hypernetwork Backbone (Feature Extraction)"]
        Conv1["Conv1 + BN + GELU<br>kernel=(3,15)"]:::layer
        Feat1["(B, 16, 12, L)"]:::tensor
        
        Conv2["Conv2 + BN + GELU + MaxPool(1,2)<br>kernel=(3,15)"]:::layer
        Feat2["(B, 32, 12, L/2)"]:::tensor
        
        Conv3["Conv3 + BN + GELU + MaxPool(1,2)<br>kernel=(3,15)"]:::layer
        Feat3["(B, 64, 12, L/4)"]:::tensor
        
        GAP["Global Average Pool<br>mean(dim=2,3)"]:::layer
        Latent["Latent Vector<br>(B, 64)"]:::tensor

        ExpInput --> Conv1 --> Feat1 --> Conv2 --> Feat2 --> Conv3 --> Feat3 --> GAP --> Latent
    end

    %% PARAMETER GENERATION HEAD
    subgraph HyperHead ["Hypernetwork Head (Linear Parameter Generation)"]
        Drop["Dropout(0.2)"]:::layer
        LinearH["Linear(64 → Classes * (M+1))<br>M = 12 * L"]:::layer
        Params["Generated Params<br>(B, Classes, M+1)"]:::tensor
        
        Split["Split into Weights & Bias"]:::layer
        GenW[("Generated Weights W<br>(B, Classes, M)")]:::param
        GenB[("Generated Bias b<br>(B, Classes)")]:::param
        
        Latent --> Drop --> LinearH --> Params --> Split
        Split --> GenW
        Split --> GenB
    end

    %% LOCAL LINEAR MODEL
    subgraph Application ["Local Linear Model Application"]
        FlatX["Flatten Input x<br>(B, M)"]:::tensor
        
        DotProd["Batched Dot Product<br>(W * x_flat)"]:::layer
        Weighted["Class-wise Scores<br>(B, Classes)"]:::tensor
        
        AddBias["Add Bias (+ b)"]:::output
        Logits[("Final Logits<br>(B, Classes)")]:::output

        Input -.-> FlatX
        GenW --> DotProd
        FlatX --> DotProd --> Weighted --> AddBias
        GenB --> AddBias --> Logits
    end

    %% Loss Functions
    subgraph Training ["Optimization Objective"]
        CE["CrossEntropyLoss"]
        L1["L1 Regularization (λ * |W|)"]
        Logits --> CE
        GenW --> L1
    end