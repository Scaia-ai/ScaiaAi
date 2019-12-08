# docsimilarity
Document Similarity

[DocSimilarity Script](./docsimilarity.py)

### running the Program
```bash
 sh run.sh
```

### Getting the Data(commands explanations on run.sh)

```bash
cd $SCAIA_HOME/docsimilarity
mkdir -p input_data/emails
python utilities/utils.py  # Will take a long time to download
cd input_data
mkdir sample
ls emails | sort -R | tail -1000 | xargs -I % cp emails/% sample # generate a random sample of 1000
export DOC_TO_TEST=`ls sample | head -1` # example: enron001_01084.txt
```



### How to Run

```bash
python docsimilarity.py <location of directory of corpus> <location of file to consider>
```


For Example:

```bash
python docsimilarity.py /path/to/my/corpus/directory/  /path/to/my/corpus/directory/abc.txt
```


Using the email data, run like this:

```bash
cd $SCAIA_HOME/docsimilarity/doc2vec
python docsimilarity.py ../input_data/sample ../input_data/sample/$DOC_TO_TEST



