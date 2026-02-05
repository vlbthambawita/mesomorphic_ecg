```mermaid
flowchart TD
    %% Style Definitions
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef layer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef tensor fill:#fff9c4,stroke:#fbc02d,stroke-width:1px,stroke-dasharray:5\,5;
    classDef param fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px;

    %% Input Signal
    Input[("Input ECG x<br>(B, 12, L)")]:::input
    ExpInput["Unsqueeze(1)<br>(B, 1, 12, L)"]:::tensor
    Input --> ExpInput

    %% ENCODER / BACKBONE
    subgraph Encoder ["Hypernetwork Backbone (Feature Extraction)"]
        direction TB
        Conv1["Conv2d(1→16)<br>kernel=(3,15)"]:::layer
        Feat1["(B, 16, 12, L)"]:::tensor
        
        Conv2["Conv2d(16→32) + MaxPool(1,2)<br>kernel=(3,15)"]:::layer
        Feat2["(B, 32, 12, L/2)"]:::tensor
        
        Conv3["Conv2d(32→64) + MaxPool(1,2)<br>kernel=(3,15)"]:::layer
        Feat3["Latent Features<br>(B, 64, 12, L/4)"]:::tensor
        
        Drop["Dropout(0.2)"]:::layer
        
        ExpInput --> Conv1 --> Feat1 --> Conv2 --> Feat2 --> Conv3 --> Feat3 --> Drop
    end

    %% WEIGHT GENERATOR
    subgraph Transition ["Transition Network (Weight Generation)"]
        direction TB
        T1["Conv2d(64→32) + Upsample(1,2)"]:::layer
        TFeat1["(B, 32, 12, L/2)"]:::tensor
        
        T2["Conv2d(32→16) + Upsample(1,2)"]:::layer
        TFeat2["(B, 16, 12, L)"]:::tensor
        
        THead["Conv2d(16→1)<br>Final Weight Projection"]:::layer
        GenW[("Generated Weights W<br>(B, 1, 12, L)")]:::param
        
        Drop --> T1 --> TFeat1 --> T2 --> TFeat2 --> THead --> GenW
    end

    %% BIAS GENERATOR
    subgraph BiasGen ["Bias Generator"]
        direction TB
        Pool["AdaptiveAvgPool2d(1,1)"]:::layer
        BFeat["(B, 64)"]:::tensor
        LinearB["Linear(64 → 1)"]:::layer
        GenB[("Generated Bias b<br>(B, 1)")]:::param
        
        Drop --> Pool --> BFeat --> LinearB --> GenB
    end

    %% LOCAL LINEAR APPLICATION
    subgraph Application ["Local Linear Model Application"]
        InputRef["Input x (Ref)<br>(B, 1, 12, L)"]:::input
        
        DotProd["Element-wise Multiply<br>(W ⊙ x)"]:::layer
        Weighted["Weighted Input<br>(B, 1, 12, L)"]:::tensor
        
        Sum["Sum over Leads & Time<br>dim=(2,3)"]:::layer
        LinearOut["(B, 1)"]:::tensor
        
        AddBias["Add Bias (+ b)"]:::output
        Logits[("Final Logits<br>(B, 1)")]:::output

        Input -.-> InputRef
        GenW --> DotProd
        InputRef --> DotProd --> Weighted --> Sum --> LinearOut
        LinearOut --> AddBias
        GenB --> AddBias --> Logits
    end

    %% Loss Functions
    subgraph Training ["Optimization Objective"]
        BCE["BCEWithLogitsLoss"]
        L1["L1 Regularization (λ * |W|)"]
        Logits --> BCE
        GenW --> L1
    end