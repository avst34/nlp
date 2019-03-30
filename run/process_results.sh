sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv goldid.goldsyn all
sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv goldid.autosyn all
sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv autoid.goldsyn all
sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv autoid.autosyn all

sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv goldid.goldsyn elmo_nosyn
sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv goldid.goldsyn elmo_syn
sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv goldid.goldsyn fasttext_nosyn
sbatch /cs/usr/aviramstern/lab/nlp_prod/run/run_process_results.sh /cs/usr/aviramstern/lab/full_model.csv /cs/usr/aviramstern/lab/full_model2.csv goldid.goldsyn fasttext_syn
