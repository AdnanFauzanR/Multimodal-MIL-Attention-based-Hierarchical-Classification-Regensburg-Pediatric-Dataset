# Pediatric Appendicitis Dataset
This repository holds the data from a cohort of pediatric patients with suspected appendicitis admitted with abdominal pain to Children’s Hospital St. Hedwig in Regensburg, Germany, between 2016 and 2021.

Each patient has (potentially multiple) ultrasound (US) images, aka views, tabular data comprising laboratory, physical examination, scoring results and ultrasonographic findings extracted manually by the experts, and three target variables, namely, diagnosis, management and severity. 

#### File Structure
Below is a brief explanation of the file structure:
* `US_Pictures/`: folder with the original B-mode ultrasound images in BMP format; images are named as `<subject #>.<view #> *.bmp`
* `app_data.xlsx`: MS Excel file with tabular data (the '*Data Summary*' tab contains explanation of the variables); *corresponding subject numbers from the `US_Pictures/` folder can be found in the column `US_Number`*
* `multiple_in_one`: a list of US image names containing multiple snapshots

#### Naming Conventions 
Following conventions are used for US image names:
| **Name** | **Meaning**               |
|----------|---------------------------|
| App      | Appendix                  |
| LN       | Lymph nodes               |
| RLQ      | Right lower quadrant      |
| FF       | Free fluids               |
| T        | Transversal               |
| L        | Longitudinal              |
| M        | Measure / Marker          |
| LAM      | Mesenterial lymphadenitis |

#### Further Reading & Maintainers

Further information about the dataset can be found in the accompanying preprint and the code repository. In case of questions, please reach out to Patricia Reis Wolfertstetter ([patricia.reiswolfertstetter@barmherzige-regensburg.de](mailto:patricia.reiswolfertstetter@barmherzige-regensburg.de)) and Ričards Marcinkevičs ([ricards.marcinkevics@inf.ethz.ch](mailto:ricards.marcinkevics@inf.ethz.ch)).

#### Copyright
This repository is copyright © 2023 Marcinkevičs, Reis Wolfertstetter, Klimiene, Ozkan, Chin-Cheong, Paschke, Zerres, Denzinger, Niederberger, Wellmann, Knorr and Vogt.

This repository is additionally licensed under CC-BY-NC-4.0.
