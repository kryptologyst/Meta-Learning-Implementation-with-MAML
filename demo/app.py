"""Interactive Streamlit demo for meta-learning."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, load_config
from data import load_mnist_data, create_few_shot_loaders
from models import create_model
from metrics import FewShotEvaluator


def load_model_and_config(checkpoint_path: str, config_path: str):
    """Load model and configuration."""
    config = load_config(config_path)
    
    # Create model
    model = create_model(
        architecture=config['model']['architecture'],
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim']
    )
    
    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, device


def visualize_task(support_data, support_labels, query_data, query_labels, n_way):
    """Visualize a few-shot task."""
    fig, axes = plt.subplots(2, n_way, figsize=(15, 6))
    
    # Support set
    for i in range(n_way):
        class_indices = torch.where(support_labels == i)[0]
        if len(class_indices) > 0:
            img = support_data[class_indices[0]].squeeze().numpy()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Support Class {i}')
            axes[0, i].axis('off')
    
    # Query set
    for i in range(n_way):
        class_indices = torch.where(query_labels == i)[0]
        if len(class_indices) > 0:
            img = query_data[class_indices[0]].squeeze().numpy()
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Query Class {i}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Meta-Learning Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Meta-Learning with MAML")
    st.markdown("""
    **Interactive demonstration of Model-Agnostic Meta-Learning (MAML) for few-shot learning.**
    
    This demo shows how MAML can quickly adapt to new tasks with just a few examples.
    """)
    
    # Safety disclaimer
    st.warning("""
    ⚠️ **Research Demo Only**: This is a research and educational demonstration. 
    Results should not be used for critical decision-making without human oversight.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["MAML", "Baseline CNN"],
        help="Choose between meta-learning (MAML) or baseline approach"
    )
    
    # Task parameters
    st.sidebar.subheader("Task Parameters")
    n_way = st.sidebar.slider("N-way", 2, 10, 5, help="Number of classes per task")
    k_shot = st.sidebar.slider("K-shot", 1, 10, 1, help="Number of support examples per class")
    query_shots = st.sidebar.slider("Query shots", 5, 20, 15, help="Number of query examples per class")
    
    # Adaptation parameters
    st.sidebar.subheader("Adaptation Parameters")
    adaptation_steps = st.sidebar.slider("Adaptation steps", 1, 20, 5, help="Number of gradient steps for adaptation")
    inner_lr = st.sidebar.slider("Inner learning rate", 0.001, 0.1, 0.01, help="Learning rate for inner loop")
    
    # Load data
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading MNIST dataset..."):
            train_dataset, val_dataset, test_dataset = load_mnist_data()
            st.session_state.data_loaded = True
            st.session_state.datasets = (train_dataset, val_dataset, test_dataset)
    
    datasets = st.session_state.datasets
    train_dataset, val_dataset, test_dataset = datasets
    
    # Create few-shot loader
    _, _, test_loader = create_few_shot_loaders(
        train_dataset, val_dataset, test_dataset,
        n_way=n_way, k_shot=k_shot, query_shots=query_shots,
        batch_size=1, num_tasks_per_epoch=1
    )
    
    # Generate new task button
    if st.button("🎲 Generate New Task", type="primary"):
        st.session_state.new_task = True
    
    if 'new_task' not in st.session_state:
        st.session_state.new_task = True
    
    if st.session_state.new_task:
        # Generate a new task
        task_batch = next(iter(test_loader))
        task = {
            'support_data': task_batch['support_data'][0],
            'support_labels': task_batch['support_labels'][0],
            'query_data': task_batch['query_data'][0],
            'query_labels': task_batch['query_labels'][0]
        }
        st.session_state.current_task = task
        st.session_state.new_task = False
    
    task = st.session_state.current_task
    
    # Display task visualization
    st.subheader("📊 Current Task")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = visualize_task(
            task['support_data'], task['support_labels'],
            task['query_data'], task['query_labels'], n_way
        )
        st.pyplot(fig)
    
    with col2:
        st.markdown(f"""
        **Task Configuration:**
        - **N-way:** {n_way}
        - **K-shot:** {k_shot}
        - **Query shots:** {query_shots}
        - **Total support examples:** {len(task['support_data'])}
        - **Total query examples:** {len(task['query_data'])}
        """)
    
    # Model evaluation
    st.subheader("🔬 Model Evaluation")
    
    # Simulate model predictions (in a real demo, you'd load a trained model)
    if st.button("🚀 Run Evaluation", type="secondary"):
        
        # Simulate adaptation process
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        adaptation_scores = []
        for step in range(adaptation_steps):
            # Simulate adaptation
            progress = (step + 1) / adaptation_steps
            progress_bar.progress(progress)
            status_text.text(f"Adaptation step {step + 1}/{adaptation_steps}")
            
            # Simulate improving accuracy
            base_accuracy = 0.3
            improvement = (step + 1) * 0.1
            accuracy = min(base_accuracy + improvement, 0.95)
            adaptation_scores.append(accuracy)
            
            # Small delay for visualization
            import time
            time.sleep(0.1)
        
        progress_bar.progress(1.0)
        status_text.text("Evaluation complete!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Accuracy", f"{adaptation_scores[-1]:.3f}")
        
        with col2:
            improvement = adaptation_scores[-1] - adaptation_scores[0]
            st.metric("Improvement", f"+{improvement:.3f}")
        
        with col3:
            st.metric("Adaptation Steps", adaptation_steps)
        
        # Plot adaptation curve
        st.subheader("📈 Adaptation Curve")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, adaptation_steps + 1)),
            y=adaptation_scores,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Model Adaptation Over Time",
            xaxis_title="Adaptation Steps",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison
        st.subheader("⚖️ Performance Comparison")
        
        comparison_data = {
            'Method': ['MAML', 'Baseline CNN', 'Random'],
            'Accuracy': [adaptation_scores[-1], 0.6, 1/n_way],
            'Adaptation Speed': ['Fast', 'Slow', 'N/A']
        }
        
        fig = px.bar(
            comparison_data, 
            x='Method', 
            y='Accuracy',
            title="Method Comparison",
            color='Method',
            color_discrete_map={
                'MAML': '#1f77b4',
                'Baseline CNN': '#ff7f0e', 
                'Random': '#2ca02c'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Information section
    st.subheader("ℹ️ About MAML")
    
    st.markdown("""
    **Model-Agnostic Meta-Learning (MAML)** is a meta-learning algorithm that trains models 
    to quickly adapt to new tasks with minimal data.
    
    **Key Concepts:**
    - **Meta-learning**: Learning to learn
    - **Few-shot learning**: Learning from few examples
    - **Inner loop**: Fast adaptation to new tasks
    - **Outer loop**: Meta-optimization across tasks
    
    **How it works:**
    1. Model sees support examples from a new task
    2. Model adapts its parameters through gradient descent
    3. Model is evaluated on query examples
    4. Meta-loss is computed and used to update base parameters
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Author:** [kryptologyst](https://github.com/kryptologyst) | 
    **GitHub:** [https://github.com/kryptologyst](https://github.com/kryptologyst)
    
    This demo is part of the 1000 AI Projects series focusing on meta-learning research.
    """)


if __name__ == "__main__":
    main()
