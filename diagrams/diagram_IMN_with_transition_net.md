# Hypernetwork-Driven Local Linear ECG Model

This document visualizes the **hypernetwork-generated local linear model** used for ECG classification.
The hypernetwork extracts latent features and generates **instance-specific weights and biases**, which are then applied to the original ECG signal.

---

## 🧠 Model Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#fff', 'primaryTextColor': '#333', 'primaryBorderColor': '#333', 'lineColor': '#555', 'secondaryColor': '#f0f0f0', 'tertiaryColor': '#fff'}}}%%
flowchart TB
    %% Color palette: Blue (input), Purple (layers), Amber (tensors), Teal (params), Coral (output)
    classDef input fill:#42A5F5,stroke:#1565C0,stroke-width:2px,color:#fff;
    classDef layer fill:#7E57C2,stroke:#4A148C,stroke-width:2px,color:#fff;
    classDef tensor fill:#FFB74D,stroke:#E65100,stroke-width:2px,color:#1a1a1a;
    classDef param fill:#26A69A,stroke:#00695C,stroke-width:2px,color:#fff;
    classDef output fill:#EF5350,stroke:#B71C1C,stroke-width:2px,color:#fff;

    %% Input
    Input[("Input Signal x<br>(B, 12, L)")]:::input
    ExpInput["Unsqueeze(1)<br>(B, 1, 12, L)"]:::tensor
    Input --> ExpInput

    %% ENCODER – compact horizontal layout
    subgraph Encoder ["Hypernetwork Backbone (Feature Extraction)"]
        direction LR
        Conv1["Conv1+BN+GELU"]:::layer
        Feat1["(16,L)"]:::tensor
        Conv2["Conv2+Pool"]:::layer
        Feat2["(32,L/2)"]:::tensor
        Conv3["Conv3+Pool"]:::layer
        Feat3["Latent<br>(64,L/4)"]:::tensor
        ExpInput --> Conv1 --> Feat1 --> Conv2 --> Feat2 --> Conv3 --> Feat3
    end

    %% BRANCH 1: WEIGHTS – compact horizontal
    subgraph Transition ["Transition Network (Weight Generation)"]
        direction LR
        Trans1["Conv+Up"]:::layer
        Trans2["Conv+Up"]:::layer
        TransHead["Conv→K"]:::layer
        GenW[("Weights W<br>(B,K,12,L)")]:::param
        Feat3 --> Trans1 --> Trans2 --> TransHead --> GenW
    end

    %% BRANCH 2: BIAS – compact horizontal
    subgraph BiasGen ["Bias Generator"]
        direction LR
        Pool["AvgPool"]:::layer
        LinearB["Linear(64→K)"]:::layer
        GenB[("Bias b<br>(B,K)")]:::param
        Feat3 --> Pool --> LinearB --> GenB
    end

    %% LOCAL LINEAR MODEL – compact horizontal
    subgraph Application ["Local Linear Model"]
        direction LR
        InputRef["x (Ref)"]:::input
        DotProd["W ⊙ x"]:::layer
        Sum["Sum"]:::layer
        AddBias["+ b"]:::output
        Logits[("Logits<br>(B,K)")]:::output
        Input -.-> InputRef
        GenW --> DotProd
        InputRef --> DotProd --> Sum --> AddBias
        GenB --> AddBias --> Logits
    end

    %% Subgraph background colors
    style Encoder fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style Transition fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    style BiasGen fill:#E0F7FA,stroke:#00838F,stroke-width:2px
    style Application fill:#FFEBEE,stroke:#C62828,stroke-width:2px

    %% Colorful thick arrows (by flow path)
    linkStyle 0 stroke:#1565C0,stroke-width:4px
    linkStyle 1,2,3,4,5,6 stroke:#7B1FA2,stroke-width:4px
    linkStyle 7,8,9,10 stroke:#6A1B9A,stroke-width:4px
    linkStyle 11,12,13 stroke:#00838F,stroke-width:4px
    linkStyle 14 stroke:#80DEEA,stroke-width:4px
    linkStyle 15 stroke:#E65100,stroke-width:4px
    linkStyle 16,17,18,19,20 stroke:#C62828,stroke-width:4px
```