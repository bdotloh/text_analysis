import contractions
import ftfy
import re 

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
import umap.plot
from hdbscan import HDBSCAN

import pandas as pd


class TextAnalysisPipeline:
    """
    Text analysis object. 
    Contains all the transformations to get from raw text to clusters. 

    On a high-level,
        preprocess --> count vectorize --> umap --> hdbscan
    
    """
    def __init__(
            self,
            cv_mindf=0.003,
            cv_ngram=(1,5), # allow for 1-letter words (like "i" and "a")
            cv_regex = r"(?u)\b\w+\b",
            umap_components=15, 
            umap_neighbours=5, 
            umap_mindist=0, 
            hdbscan_minsample=5
            ):
        
        # initialise transformations 
        self.cv=CountVectorizer(
            ngram_range=cv_ngram,
            min_df=cv_mindf,
            token_pattern=cv_regex
        )

        self.encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.umap=UMAP(
            n_components=umap_components,
            min_dist=umap_mindist,
            n_neighbors=umap_neighbours
        )

        self.clusterer = HDBSCAN(min_samples=hdbscan_minsample)
    
    def run(self, documents,indices=None):
        self.dtm=self._createDTM(documents,indices)

        terms=self.dtm.columns
        embeddings=self.encoder.encode(terms,show_progress_bar=True)
        embeddings_reduced=self.umap.fit_transform(embeddings)
        
        clusters=self.clusterer.fit_predict(embeddings_reduced)
        term_cluster=pd.DataFrame({'cluster':clusters},index=terms)
        
        # concat cluster terms, such that each row contains a list of terms, with their respective cluster value
        terms_cluster=pd.Series(index=term_cluster.cluster.value_counts().index)
        terms_cluster.sort_index(inplace=True)
        
        for i in terms_cluster.index:
            terms_cluster[i]= ', '.join(term_cluster[term_cluster.cluster==i].index)
        
        p = self.vizclusters(term_cluster)

        return term_cluster,terms_cluster,p
        
    def vizclusters(self,term_cluster_df,pointsize=3.5,ht=640,wt=640):
        """
        Project n-gram embeddings onto a 2-d space, use cluster labels to color each point.
        Assumes that:
            1) 'dtm' is a n*d df with where n = num docs and d= terms.
            2) 'cluster' is a n-dimensional column vector
        """
        
        embs=self.encoder.encode(term_cluster_df.index.tolist())  
        umapper=UMAP(n_components=2).fit(embs)
    

        p = umap.plot.interactive(
            umapper,
            labels = term_cluster_df.cluster.values,
            hover_data = term_cluster_df.reset_index(), 
            point_size=pointsize, 
            background='black',
            color_key_cmap='Paired',
            height=ht,
            width=wt
            )
        
        return p

    def _preprocess(self, documents):
        ### text clean up
        def rmlinks(text): # remove hyperlinks and emails
            return re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', str(text))
        out = rmlinks(documents)
        out = out.replace("’", "'").replace("…", "...").lower()
        out = contractions.fix(ftfy.fix_text(out))
        out = ' '.join(word_tokenize(out))   
        
        return out
    
    def _createDTM(self,docs,indices):
        cleaned_docs=list(map(self._preprocess,docs))
        dtm=pd.DataFrame.sparse.from_spmatrix(self.cv.fit_transform(cleaned_docs))
        dtm.columns=self.cv.get_feature_names_out()
        dtm.index=indices
        return dtm
