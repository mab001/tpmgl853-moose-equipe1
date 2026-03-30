import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless mode for CI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

 
# Lire le fichier CSV exporté par Moose (séparateur ;)
df = pd.read_csv('export_metrics.csv', sep=';')

# Renommer les colonnes pour uniformiser
df = df.rename(columns={
    'Nom_Classe': 'ClassName',
    'Nb_Methodes': 'NumberOfMethods',
    'Nb_Attributs': 'NumberOfAttributes',
    'Lignes_de_Code': 'LinesOfCode',
    'CC_Total': 'CyclomaticTotal',
    'CC_Avg': 'CyclomaticAvg',
    'LOC_NOM': 'LocPerMethod'
})

# Lire le fichier method-level complexity (si disponible)
try:
    df_methods = pd.read_csv('export_methods_complexity.csv', sep=';')
    print(f"✓ Methods complexity data loaded: {len(df_methods)} methods")
except FileNotFoundError:
    print("⚠️  export_methods_complexity.csv not found")

# Afficher les données
print(df.head(10))
print(f"\nDimensions: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Colonnes: {df.columns.tolist()}")

# ===== GRAPHIQUE 1: COMPARAISON DES MÉTRIQUES =====
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(14, 8))

categories = df['ClassName'].tolist()
methods = df['NumberOfMethods'].tolist()
attributes = df['NumberOfAttributes'].tolist()
lines_of_code = df['LinesOfCode'].tolist()

x = np.arange(len(categories))
width = 0.25

bars1 = ax.bar(x - width, methods, width, label='Nombre de méthodes', color='#3B82F6', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, attributes, width, label='Nombre d\'attributs', color='#10B981', edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, lines_of_code, width, label='Lignes de code', color='#EF4444', edgecolor='black', linewidth=1.2)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
ax.set_ylabel('Valeur', fontsize=12, fontweight='bold')
ax.set_title('Comparaison des métriques par classe', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.set_facecolor('#F3F4F6')
fig.patch.set_facecolor('white')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('graphique_comparaison_metriques.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique comparaison sauvegardé: graphique_comparaison_metriques.png")

# ===== GRAPHIQUE 2: RADAR CHART =====

categories_metrics = ['Méthodes', 'Attributs', 'Lignes de code']
num_vars = len(categories_metrics)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

num_classes = len(df)
fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5), subplot_kw=dict(projection='polar'))

if num_classes == 1:
    axes = [axes]

colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

for idx, class_name in enumerate(df['ClassName']):
    ax = axes[idx]

    values = [
        df.loc[idx, 'NumberOfMethods'],
        df.loc[idx, 'NumberOfAttributes'],
        df.loc[idx, 'LinesOfCode']
    ]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx])
    ax.fill(angles, values, alpha=0.25, color=colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_metrics, fontsize=10)
    ax.set_ylim(0, max(df['LinesOfCode'].max(), df['NumberOfMethods'].max()) * 1.1)
    ax.set_title(f'{class_name}', fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

plt.tight_layout()
plt.savefig('graphique_radar_metriques.png', dpi=300, bbox_inches='tight')
print("✓ Graphique radar sauvegardé: graphique_radar_metriques.png")

# ===== GRAPHIQUE 3: HEATMAP =====

fig, ax = plt.subplots(figsize=(10, 6))

data_for_heatmap = df[['ClassName', 'NumberOfMethods', 'NumberOfAttributes', 'LinesOfCode']].set_index('ClassName')

sns.heatmap(data_for_heatmap.T, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Valeur'}, ax=ax, linewidths=2)

ax.set_title('Heatmap des métriques par classe', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
ax.set_ylabel('Métriques', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('graphique_heatmap_metriques.png', dpi=300, bbox_inches='tight')
print("✓ Graphique heatmap sauvegardé: graphique_heatmap_metriques.png")

# ===== GRAPHIQUE 4: COMPLEXITÉ VS TAILLE =====

fig, ax = plt.subplots(figsize=(12, 8))

x_vals = df['LinesOfCode'].tolist()
y_vals = df['NumberOfMethods'].tolist()
sizes = [max(attr, 1) * 100 for attr in df['NumberOfAttributes'].tolist()]

scatter = ax.scatter(x_vals, y_vals, s=sizes, c=range(len(categories)),
                     cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)

for i, cat in enumerate(categories):
    ax.annotate(cat,
                (x_vals[i], y_vals[i]),
                fontsize=11,
                fontweight='bold',
                ha='right',
                xytext=(-10, 10),
                textcoords='offset points')

ax.set_xlabel('Lignes de code (LOC)', fontsize=12, fontweight='bold')
ax.set_ylabel('Nombre de méthodes (Complexité)', fontsize=12, fontweight='bold')
ax.set_title('Complexité vs Taille des classes', fontsize=14, fontweight='bold', pad=20)
ax.set_facecolor('#F9FAFB')
fig.patch.set_facecolor('white')
ax.grid(True, alpha=0.3, linestyle='--')

if max(y_vals) > 0:
    ax.axhspan(max(y_vals) * 0.7, max(y_vals) * 1.2, alpha=0.1, color='red', label='Zone problématique')

for size, label in zip([2, 4, 6], ['2 attributs', '4 attributs', '6 attributs']):
    ax.scatter([], [], s=size*100, c='gray', alpha=0.6, edgecolors='black', label=label)
ax.legend(scatterpoints=1, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('graphique_complexite_vs_taille.png', dpi=300, bbox_inches='tight')
print("✓ Graphique complexité vs taille sauvegardé: graphique_complexite_vs_taille.png")

# ===== GRAPHIQUE 5: COMPLEXITÉ DES MÉTHODES PAR CLASSE =====

if 'df_methods' in locals():
    # Get top 15 most complex methods
    df_methods_top = df_methods.nlargest(15, 'Complexity')[['ClassName', 'MethodName', 'Complexity', 'LOC']].copy()
    df_methods_top['Label'] = df_methods_top['ClassName'].astype(str) + '.' + df_methods_top['MethodName'].astype(str)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors_by_class = {}
    class_palette = sns.color_palette('husl', len(df_methods_top['ClassName'].unique()))
    for idx, class_name in enumerate(df_methods_top['ClassName'].unique()):
        colors_by_class[class_name] = class_palette[idx]
    
    bar_colors = [colors_by_class[cn] for cn in df_methods_top['ClassName']]
    
    bars = ax.barh(range(len(df_methods_top)), df_methods_top['Complexity'], color=bar_colors, edgecolor='black', linewidth=1.2)
    
    for i, (idx, row) in enumerate(df_methods_top.iterrows()):
        ax.text(row['Complexity'] + 0.1, i, f'CC={int(row["Complexity"])} LOC={int(row["LOC"])}', 
                va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(df_methods_top)))
    ax.set_yticklabels(df_methods_top['Label'], fontsize=10)
    ax.set_xlabel('Cyclomatic Complexity', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 des méthodes les plus complexes', fontsize=16, fontweight='bold', pad=20)
    ax.set_facecolor('#F9FAFB')
    fig.patch.set_facecolor('white')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend for classes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_by_class[cn], edgecolor='black', label=cn) 
                       for cn in sorted(colors_by_class.keys())]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, title='Classe')
    
    plt.tight_layout()
    plt.savefig('graphique_complexite_methodes.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique complexité des méthodes sauvegardé: graphique_complexite_methodes.png")
    
    # ===== SOUS-GRAPHIQUE: COMPLEXITÉ PAR CLASSE =====
    
    fig, axes = plt.subplots(1, len(df), figsize=(6 * len(df), 5))
    if len(df) == 1:
        axes = [axes]
    
    for ax_idx, class_name in enumerate(df['ClassName']):
        ax = axes[ax_idx]
        
        # Filter methods for this class
        class_methods = df_methods[df_methods['ClassName'] == class_name].copy()
        class_methods = class_methods.sort_values('Complexity', ascending=True)
        
        if len(class_methods) > 0:
            colors_complexity = ['#EF4444' if cc > 5 else '#F59E0B' if cc > 2 else '#10B981' 
                                for cc in class_methods['Complexity']]
            
            bars = ax.barh(range(len(class_methods)), class_methods['Complexity'], 
                           color=colors_complexity, edgecolor='black', linewidth=1)
            
            for i, (idx, row) in enumerate(class_methods.iterrows()):
                ax.text(row['Complexity'] + 0.05, i, f'{int(row["Complexity"])}', 
                        va='center', fontsize=8, fontweight='bold')
            
            ax.set_yticks(range(len(class_methods)))
            ax.set_yticklabels(class_methods['MethodName'], fontsize=9)
            ax.set_xlabel('Cyclomatic Complexity', fontsize=10, fontweight='bold')
            ax.set_title(f'{class_name}\n({len(class_methods)} méthodes)', fontsize=12, fontweight='bold')
            ax.set_facecolor('#F9FAFB')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_xlim(0, max(class_methods['Complexity'].max() * 1.2, 2))
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('graphique_complexite_par_classe.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique complexité par classe sauvegardé: graphique_complexite_par_classe.png")

print("\n✓ Tous les graphiques ont été générés en PNG!")
