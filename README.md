# Sustainable Supply Chain Optimization using Gurobi

This project presents a **multi-objective optimization model** for a sustainable supply chain considering **economic, environmental, and social** dimensions. It is implemented in Python using **Gurobi Optimizer** and compares five widely used multi-objective techniques.

---

## 📌 Problem Overview

We aim to determine the optimal production quantities for multiple products, balancing:

- 💰 **Profit maximization** (economic)
- 🌱 **Emission minimization** (environmental)
- 👷 **Labor maximization** (social)

Subject to constraints like:
- Production capacity
- Budget
- Emission caps
- Demand limits
- Minimum labor requirement

---

## ⚙️ Methods Implemented

The following multi-objective optimization methods are implemented and compared:

1. **Aggregated Method**
2. **Weighted Sum Method** (with normalization)
3. **ε-Constrained Method**
4. **Lexicographic Method**
5. **Goal Programming**

---


---

## 🔧 Dependencies

- Python ≥ 3.8
- [Gurobi Optimizer](https://www.gurobi.com/)
- `gurobipy` Python API

### Installation

```bash
# Activate your environment (if using one)
source ~/gurobi-env/bin/activate

# Install Gurobi Python bindings
pip install gurobipy

