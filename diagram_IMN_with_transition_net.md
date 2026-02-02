# Hypernetwork-Driven Local Linear ECG Model

This document visualizes the **hypernetwork-generated local linear model** used for ECG classification.
The hypernetwork extracts latent features and generates **instance-specific weights and biases**, which are then applied to the original ECG signal.

---

## 🧠 Model Architecture

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
        Feat3["Latent Features<br>(B, 64, 12, L/4)"]:::tensor
        
        ExpInput --> Conv1 --> Feat1 --> Conv2 --> Feat2 --> Conv3 --> Feat3
    end

    %% BRANCH 1: WEIGHTS
    subgraph Transition ["Transition Network (Weight Generation)"]
        Trans1["Conv(64→32) + Upsample(1,2)"]:::layer
        TFeat1["(B, 32, 12, L/2)"]:::tensor
        
        Trans2["Conv(32→16) + Upsample(1,2)"]:::layer
        TFeat2["(B, 16, 12, L)"]:::tensor
        
        TransHead["Conv(16→K)<br>Linear Projection"]:::layer
        GenW[("Generated Weights W<br>(B, K, 12, L)")]:::param
        
        Feat3 --> Trans1 --> TFeat1 --> Trans2 --> TFeat2 --> TransHead --> GenW
    end

    %% BRANCH 2: BIAS
    subgraph BiasGen ["Bias Generator"]
        Pool["AdaptiveAvgPool(1,1)"]:::layer
        BFeat["(B, 64, 1, 1) → (B, 64)"]:::tensor
        
        LinearB["Linear(64 → K)"]:::layer
        GenB[("Generated Bias b<br>(B, K)")]:::param
        
        Feat3 --> Pool --> BFeat --> LinearB --> GenB
    end

    %% LOCAL LINEAR MODEL
    subgraph Application ["Local Linear Model Application"]
        InputRef["Input x (Ref)<br>(B, 1, 12, L)"]:::input
        Input -.-> InputRef
        
        DotProd["Element-wise Multiply<br>(W ⊙ x)"]:::layer
        Weighted["Weighted Input<br>(B, K, 12, L)"]:::tensor
        
        Sum["Sum over dims (2,3)<br>(Leads & Time)"]:::layer
        LinearOut["(B, K)"]:::tensor
        
        AddBias["Add Bias (+ b)"]:::output
        Logits[("Final Logits<br>(B, K)")]:::output

        GenW --> DotProd
        InputRef --> DotProd --> Weighted --> Sum --> LinearOut
        LinearOut --> AddBias
        GenB --> AddBias --> Logits
    end