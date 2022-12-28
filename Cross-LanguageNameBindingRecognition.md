# Cross-LanguageNameBindingRecognition

![0912](E:\nuaa\1st_Year\3_presentation\drawio\0912.png)

## DataPreprocessing

| Project    | Domain                               | Framework                 | Size    | LOC  | All field | Correct name binding pair | Incorrect name binding pair |
| ---------- | ------------------------------------ | ------------------------- | ------- | ---- | --------- | ------------------------- | --------------------------- |
| itracker   | Issue Tracker                        | Spring,Hibernate          | 10940kB | 110k | 773       | 2014                      | 94220                       |
| sagan      | Reference Application                | Spring                    | 6547kB  | 20k  | 323       | 470                       | 27892                       |
| springside | JavaEE Application Reference Example | Spring                    | 2038kB  | 28k  | 287       | 72                        | 15542                       |
| Tudu-Lists | Todo Lsts Management                 | Spring,Hibernate          | 822kB   | 8k   | 76        | 92                        | 2047                        |
| zksample2  | ZK Framework Application             | Spring,Hibernate          | 5262kB  | 35k  | 1067      | 402                       | 15674                       |
| jrecruiter | Job Posting Solutions                | Spring,Hibernate,Struts   | 7327kB  | 13k  | 163       | 112                       | 8519                        |
| hispacta   | Maven Web Application                | Spring,Hibernate,Tapestry | 183kB   | 3k   | 68        | 154                       | 1719                        |
| powerstone | Java Workflow Management System      | Spring,Hibernate          | 2448kB  | 31k  | 371       | 592                       | 8760                        |
| jtrac      | Issue Tracker                        | Spring,Hibernate,Wicket   | 2149kB  | 22k  | 328       | 284                       | 20604                       |
| mall       | E-commerce System                    | Spring,SpringBoot,MyBatis | 9103kB  | 87k  | 600       | 8151                      | 294290                      |

## NameBindingRecognitionModelTraining

1. *BERT fine-tuning model*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/blob/main/NameBindingRecognitionModelTraining/run1.py

## DuplicateNameBindingDiscrimination

1. *Context information collection:*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/tree/main/DuplicateNameBindingDiscrimination/gettoken

2. *Feature extraction*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/tree/main/DuplicateNameBindingDiscrimination/feature

3. *Duplicate-name discrimination model training*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/blob/main/DuplicateNameBindingDiscrimination/classifiers.py