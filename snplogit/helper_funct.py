import random

def get_snp_info(rsid):
    # Ensembl REST API endpoint for SNPs
    url = f"https://rest.ensembl.org/variation/human/{rsid}?"
    headers = {"Content-Type": "application/json"}
    
    # Send GET request to Ensembl API
    response = requests.get(url, headers=headers)
    
    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        # Extract chromosome and position
        if 'mappings' in data:
            chromosome = data['mappings'][0]['seq_region_name']
            position = data['mappings'][0]['start']
            return chromosome, position
        else:
            print("SNP mappings not found in response.")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None
    
def plot_manhattan(data, threshold=1.3, highlight_snp=None):
    # Set up color scheme for chromosomes
    colors = ['darkred','darkgreen','darkblue', 'gold']  # Two colors for alternating chromosomes
    
    # Sort data by chromosome and position
    data.CHR = data.CHR.astype(int)
    data = data.sort_values(['CHR', 'BP'])
    data['ind'] = range(len(data))  # Unique position identifier for plotting

    # Create the Manhattan plot
    plt.figure(figsize=(10, 5.75))
    
    # Plot each chromosome with alternating colors
    x_labels_pos = []
    x_labels = []
    for i, (chromosome, group) in enumerate(data.groupby('CHR')):
        plt.scatter(group['ind'], group['-log10(P)'],
                    color=colors[i % len(colors)], s=15, label=f'Chr {chromosome}')
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
        x_labels.append(chromosome)
    
    # Highlight specific SNPs if provided
    if highlight_snp:
        highlight = data[data['SNP'].isin(highlight_snp)]
        plt.scatter(highlight['ind'], highlight['-log10(P)'], color='red', s=20)
    
    # Draw genome-wide significance threshold line
    plt.axhline(y=threshold, color='grey', linestyle='--', linewidth=1)
    
    # Plot customization
    plt.xlabel('Chromosome')
    plt.ylabel('-log10(P)')
    plt.ylim(0,4)
    plt.title('Manhattan Plot')
    plt.xticks(x_labels_pos, x_labels)  # Removing x-axis ticks (optional)
    plt.legend(title="Chromosomes", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def is_ordinal(data, feature_name, target=None, correlation_threshold=0.5):
    """
    Automatically detect if a categorical feature is ordinal.
    - Checks if categories follow a logical ordering or have a monotonic correlation with a numerical target.
    - If no target is provided, checks for the frequency distribution of the feature.
    """
    
    feature = data[feature_name]
    
    # 1. Check if categories follow a monotonic relationship with a numerical target (if provided)
    if target is not None:
        le = LabelEncoder()
        encoded = le.fit_transform(feature)
        
        # Compute Spearman correlation to check for monotonic relationship
        correlation, _ = spearmanr(encoded, target)
        
        # If the correlation is strong (positive or negative), consider it ordinal
        if abs(correlation) > correlation_threshold:
            print(f"Feature '{feature_name}' is likely ordinal due to monotonic relationship with the target (Spearman correlation: {correlation:.2f})")
            return True
    
    # 2. Check if the categories have a frequency distribution that suggests order
    freq_dist = feature.value_counts().sort_values()
    print(f"Frequency distribution of '{feature_name}':\n{freq_dist}\n")
    
    # If the frequency distribution is sorted (or nearly sorted), it could indicate ordinal data
    # Simple heuristic: check if the sorted order of frequencies is consistent
    if all(freq_dist.index[i] < freq_dist.index[i + 1] for i in range(len(freq_dist) - 1)):
        print(f"Feature '{feature_name}' is likely ordinal based on frequency distribution.")
        return True

    # 3. If none of the above checks provide conclusive evidence, assume it's not ordinal
    print(f"Feature '{feature_name}' is likely not ordinal.")
    return False
    
def generate_random_color():
    rand = lambda: random.randint(100, 255)
    return '#%02X%02X%02X' % (rand(), rand(), rand())