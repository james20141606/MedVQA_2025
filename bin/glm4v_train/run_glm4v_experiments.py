#!/usr/bin/env python3
"""
Master script to run GLM-4.1V experiments on MedVQA datasets
"""
import os
import subprocess
import argparse
import time


def submit_job(script_path, job_name, dependencies=None):
    """Submit SLURM job and return job ID"""
    cmd = ["sbatch"]
    
    if dependencies:
        dep_str = ":".join(str(dep) for dep in dependencies)
        cmd.extend(["--dependency", f"afterok:{dep_str}"])
    
    cmd.append(script_path)
    
    print(f"Submitting job: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Extract job ID from output like "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted {job_name} with job ID: {job_id}")
        return int(job_id)
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit {job_name}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run GLM-4.1V experiments on MedVQA datasets')
    parser.add_argument('--mode', choices=['individual', 'combined', 'both'], default='both',
                        help='Training mode: individual models, combined model, or both')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation (skip training)')
    parser.add_argument('--jobs_dir', default='/home/xc1490/xc1490/projects/medvqa_2025/jobs',
                        help='Directory containing SLURM job scripts')
    
    args = parser.parse_args()
    
    job_ids = []
    
    if not args.eval_only:
        # Submit training jobs
        if args.mode in ['individual', 'both']:
            print("Submitting individual training jobs...")
            individual_job_id = submit_job(
                os.path.join(args.jobs_dir, 'glm4v_individual_train.sbatch'),
                'individual_training'
            )
            if individual_job_id:
                job_ids.append(individual_job_id)
        
        if args.mode in ['combined', 'both']:
            print("Submitting combined training job...")
            combined_job_id = submit_job(
                os.path.join(args.jobs_dir, 'glm4v_combined_train.sbatch'),
                'combined_training'
            )
            if combined_job_id:
                job_ids.append(combined_job_id)
        
        # Wait a bit before submitting evaluation jobs
        if job_ids:
            print(f"Waiting for training jobs to start before submitting evaluation...")
            time.sleep(5)
    
    # Submit evaluation jobs (dependent on training completion)
    print("Submitting evaluation jobs...")
    eval_dependencies = job_ids if job_ids else None
    
    eval_job_id = submit_job(
        os.path.join(args.jobs_dir, 'glm4v_evaluate.sbatch'),
        'evaluation',
        dependencies=eval_dependencies
    )
    
    if eval_job_id:
        job_ids.append(eval_job_id)
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUBMISSION SUMMARY")
    print("="*50)
    
    if job_ids:
        print(f"Successfully submitted {len(job_ids)} job(s):")
        for i, job_id in enumerate(job_ids, 1):
            print(f"  {i}. Job ID: {job_id}")
        
        print(f"\nTo monitor jobs: squeue -u $USER")
        print(f"To cancel all jobs: scancel {' '.join(str(jid) for jid in job_ids)}")
    else:
        print("No jobs were submitted successfully.")
    
    print("="*50)


if __name__ == '__main__':
    main()
