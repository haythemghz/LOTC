import yaml
import os, glob
import numpy as np

# Custom loader to handle numpy scalars if they weren't saved with the standard yaml format
# But usually PyYAML can handle them if numpy is imported or if we use a safe loader that allows them.
# The error was about !!python/object/apply:numpy.core.multiarray.scalar

def handle_numpy(loader, node):
    return node.value

yaml.add_multi_constructor('!!python/object/apply:numpy', lambda loader, suffix, node: None)

# Read all available rigorous results
for f in sorted(glob.glob('experiments/results/*_rigorous.yaml')):
    print(f"\n=== {os.path.basename(f)} ===")
    try:
        with open(f, 'r') as stream:
            # We use UnsafeLoader or FullLoader if we trust the source to handle the custom objects
            # Or we can just ignore the complex objects and get the values
            d = yaml.load(stream, Loader=yaml.Loader)
    except Exception as e:
        print(f"Error loading {f}: {e}")
        continue
        
    if 'stats_vs_kmeans' not in d:
        continue
        
    s = d['stats_vs_kmeans']
    for k in ['ARI','ACC','NMI']:
        if k not in s: continue
        info = s[k]
        # Ensure values are floats for printing
        ml = float(info['mean_lotc'])
        sl = float(info['std_lotc'])
        mb = float(info['mean_base'])
        sb = float(info['std_base'])
        pv = float(info['p_value'])
        cd = float(info['cohens_d'])
        ci = [float(info['ci_95'][0]), float(info['ci_95'][1])]
        
        print(f"  {k}: LOTC={ml:.4f} +/- {sl:.4f}  KM={mb:.4f} +/- {sb:.4f}  p={pv:.4e}  d={cd:.3f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
