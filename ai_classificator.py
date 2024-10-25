import ollama
import pandas as pd


qwen = 'qwen2.5-coder:7b'
mistral_nemo = "mistral-nemo:12b"

# response = ollama.chat(model=mistral_nemo, messages=[
# {
# 'role': 'user',
# 'content': 'Why is the sky blue?',
# },
# ])
#
# print(response['message']['content'])


jobGroups = [
    "IT и технологии",
    "Медицина и фармацевтика",
    "Финансы и бухгалтерия",
    "Менеджмент и руководство",
    "Связь с общественностью",
    "Производство и строительство",
    "Торговля",
    "Образование и наука",
    "Общественный сектор"
]

baseTask = "You have to assign a group to the job offer only by knowing it's name. ONLY answer with a group name AND you can ONLY PICK A GROUP FROM job_groups. "




ekat_data = pd.read_csv('ekat_jobs.csv')
perm_data = pd.read_csv('perm_jobs.csv')
df_ekat = pd.DataFrame(ekat_data)
df_perm = pd.DataFrame(perm_data)


dfs = [('ekat', df_ekat), ('perm', df_perm)]

for (dataset_name, df) in dfs:
    print(f"\n{dataset_name} dataset:\n")
    for index, row in df.iterrows():
        job_offer_name = row['name']
        promptik = f"{baseTask}DATA: job_offer_name={job_offer_name} \njob_groups={', '.join(jobGroups)}"
        response = ollama.generate(model=mistral_nemo, prompt=promptik)
        print(f"{job_offer_name}: {response['response']}")
