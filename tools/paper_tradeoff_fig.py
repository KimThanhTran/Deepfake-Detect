"""Grouped bar chart: mean Real-Acc vs Fake-Acc of the 6 configurations (a)-(f),
illustrating the robustness<->sensitivity tradeoff driven by augmentation and how
score fusion keeps the baseline's Fake-Acc. Means over the 5 benchmarks (ABLATION_FINAL)."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

configs = ['(a) NPR\nbaseline', '(b) NPR\n+ Aug (mạnh)', '(c) AdaNPR\n+ Aug (mạnh)', '(d) AdaNPR\n+ Aug (nhẹ)',
           '(e) AdaNPR\n(không Aug)', '(f) Kết hợp\nđiểm (a)+(c)']
real = [76.93, 81.81, 86.68, 90.74, 78.44, 77.92]
fake = [93.04, 74.08, 74.32, 61.95, 70.97, 93.35]

x = np.arange(len(configs))
w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 4.2))
b1 = ax.bar(x - w/2, real, w, label='Real-Acc (ảnh thật)', color='#2f6fb0')
b2 = ax.bar(x + w/2, fake, w, label='Fake-Acc (ảnh giả)', color='#c0504d')
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.6,
                f'{r.get_height():.1f}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Độ chính xác trung bình (%)', fontsize=11)
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.get_xticklabels()[-1].set_fontweight('bold')
ax.legend(fontsize=10, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else 'results/fig_tradeoff.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('Saved', out)
