"""
RLHF & Data Annotation Metrics Module
Calculates inter-annotator agreement and response quality scores.
"""

from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np


def calculate_cohen_kappa(annotator_a: list, annotator_b: list) -> dict:
    """
    Calculate Cohen's Kappa for inter-annotator agreement.
    
    Args:
        annotator_a: List of labels from first annotator
        annotator_b: List of labels from second annotator
        
    Returns:
        Dictionary with kappa score, interpretation and raw accuracy
    """
    if len(annotator_a) != len(annotator_b):
        raise ValueError("Both annotators must have same number of annotations")
    
    kappa = cohen_kappa_score(annotator_a, annotator_b)
    accuracy = accuracy_score(annotator_a, annotator_b)
    
    # Interpretation scale (Landis & Koch, 1977)
    if kappa < 0:
        interpretation = "Poor agreement"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    return {
        "kappa": round(kappa, 4),
        "accuracy": round(accuracy, 4),
        "interpretation": interpretation,
        "n_samples": len(annotator_a)
    }


def calculate_preference_score(preferences: list) -> dict:
    """
    Calculate preference distribution for RLHF ranking tasks.
    
    Args:
        preferences: List of strings ('A', 'B', or 'Tie')
        
    Returns:
        Dictionary with preference percentages
    """
    total = len(preferences)
    counts = {
        'A': preferences.count('A'),
        'B': preferences.count('B'),
        'Tie': preferences.count('Tie')
    }
    
    return {
        "total_annotations": total,
        "preferences": {k: round(v/total*100, 2) for k, v in counts.items()},
        "win_rate_A": round(counts['A']/total*100, 2),
        "win_rate_B": round(counts['B']/total*100, 2)
    }


def generate_annotator_report(annotator_a: list, annotator_b: list, 
                              preferences: list = None) -> str:
    """
    Generate a formatted report combining all metrics.
    """
    agreement = calculate_cohen_kappa(annotator_a, annotator_b)
    
    report = f"""
══════════════════════════════════════════════╗
║     RLHF ANNOTATION QUALITY REPORT          ║
══════════════════════════════════════════════╣
║ Inter-Annotator Agreement                   ║
║   Cohen's Kappa: {agreement['kappa']:<20} ║
║   Raw Accuracy:  {agreement['accuracy']:<20} ║
║   Interpretation:{agreement['interpretation']:<19} ║
║   Samples:       {agreement['n_samples']:<20} ║
╠══════════════════════════════════════════════╣
"""
    
    if preferences:
        pref = calculate_preference_score(preferences)
        report += f"""║ Preference Distribution                 ║
║   Model A Win Rate: {pref['win_rate_A']:<17}% ║
║   Model B Win Rate: {pref['win_rate_B']:<17}% ║
║   Ties:             {pref['preferences']['Tie']:<17}% ║
"""
    
    report += "══════════════════════════════════════════════╝"
    return report