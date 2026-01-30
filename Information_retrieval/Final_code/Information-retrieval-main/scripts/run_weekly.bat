@echo off
python -m search_engine.crawler --seed "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo/publications/" --org "CSM" --max-pages 500 --delay 1.0
