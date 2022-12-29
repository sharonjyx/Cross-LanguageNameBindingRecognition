# Cross-LanguageNameBindingRecognition

We propose a cross-language name binding approach for the Java frameworks based on the deep learning model. This novel approach determines the name binding by string matching, combining context information, and framework fundamental rules.

## DataPreprocessing
https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/tree/main/DataPreprocessing

We manually construct a benchmark dataset containing 10 open source projects programmed by Java frameworks with multiple languages. If you want to build it manually, you can gather all the project fields on your own, create name binding pairs using string global matching, gather the necessary data, and mark it manually for identification. For a detailed description of the steps, see the essay' section on experimental setup's data collection.

| Project    | Framework                 | LOC  | All field | Correct name binding pair | Incorrect name binding pair |
| ---------- | :------------------------ | ---- | --------- | ------------------------- | --------------------------- |
| itracker   | Spring,Hibernate          | 110k | 773       | 2014                      | 94220                       |
| sagan      | Spring                    | 20k  | 323       | 470                       | 27892                       |
| springside | Spring                    | 28k  | 287       | 72                        | 15542                       |
| Tudu-Lists | Spring,Hibernate          | 8k   | 76        | 92                        | 2047                        |
| zksample2  | Spring,Hibernate          | 35k  | 1067      | 402                       | 15674                       |
| jrecruiter | Spring,Hibernate,Struts   | 13k  | 163       | 112                       | 8519                        |
| hispacta   | Spring,Hibernate,Tapestry | 3k   | 68        | 154                       | 1719                        |
| powerstone | Spring,Hibernate          | 31k  | 371       | 592                       | 8760                        |
| jtrac      | Spring,Hibernate,Wicket   | 22k  | 328       | 284                       | 20604                       |
| mall       | Spring,SpringBoot,MyBatis | 87k  | 600       | 8151                      | 294290                      |

- uniquematch.csv——Name binding dataset with no duplicate fields. (A:Java code B: non-Java code C:class label)
- repeatmatch.csv——Name binding dataset with duplicate fields. (A:Java code B: non-Java code C:class label)
- repeatmatchless.csv——Name binding dataset with duplicate fields. Binding pairs that are certain to hold are positive cases. Negative examples include both the Java and non-Java components of the valid binding pair that were not matched by field. (A:field B:Java code C:non-Java code D:class label E:Java filename F:non-Java filename G:non-Java filename H:Java filepath I:non-Java filepath)
- repeattoken.csv——Name binding dataset with duplicate fields. Binding pairs that are certain to hold are positive cases. Negative examples include both the Java and non-Java components of the valid binding pair that were not matched by field. (A:field B:Java context C:non-Java context D:class label E:Java filename F:non-Java filename G:non-Java filename H:Java filepath I:non-Java filepath)
- The remaining files are the intermediate files for data processing, the datasets used in the comparison experiments, and the balanced datasets

## NameBindingRecognitionModelTraining

1. *BERT fine-tuning model*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/blob/main/NameBindingRecognitionModelTraining/run1.py

## DuplicateNameBindingDiscrimination

1. *Context information collection:*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/tree/main/DuplicateNameBindingDiscrimination/gettoken

2. *Feature extraction——cosine_sim.csv + fieldsrelationship.csv + relationship.csv*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/tree/main/DuplicateNameBindingDiscrimination/feature

3. *Duplicate-name discrimination model training*

   https://github.com/sharonjyx/Cross-LanguageNameBindingRecognition/blob/main/DuplicateNameBindingDiscrimination/classifiers.py
