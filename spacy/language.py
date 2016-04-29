from __future__ import absolute_import
from os import path
from warnings import warn
import io

try:
    import ujson as json
except ImportError:
    import json

from .tokenizer import Tokenizer
from .vocab import Vocab
from .syntax.parser import Parser
from .tagger import Tagger
from .matcher import Matcher
from .serialize.packer import Packer
from . import attrs
from . import orth

from . import util
from . import about
from .attrs import TAG, DEP, ENT_IOB, ENT_TYPE, HEAD


class Language(object):
    lang = None

    Morphology = morphology.Morphology
    Vocab = vocab.Vocab
    Vectors = vectors.Vectors
    Tokenizer = tokenizer.Tokenizer

    lex_attrs = {
        attrs.LOWER: orth.lower,
        attrs.NORM: orth.norm,
        attrs.SHAPE: orth.word_shape,
        attrs.PREFIX: orth.prefix,
        attrs.SUFFIX: orth.suffix,
        attrs.CLUSTER: orth.cluster,
        attrs.PROB: lambda string: -20,
        attrs.IS_ALPHA: orth.is_alpha,
        attrs.IS_ASCII: orth.is_ascii,
        attrs.IS_DIGIT: cls.is_digit,
        attrs.IS_LOWER: orth.is_lower,
        attrs.IS_PUNCT: orth.is_punct,
        attrs.IS_SPACE: orth.is_space,
        attrs.IS_TITLE: orth.is_title,
        attrs.IS_UPPER: orth.is_upper,
        attrs.IS_BRACKET: orth.is_bracket,
        attrs.IS_QUOTE: orth.is_quote,
        attrs.IS_LEFT_PUNCT: orth.is_left_punct,
        attrs.IS_RIGHT_PUNCT: orth.is_right_punct,
        attrs.LIKE_URL: orth.like_url,
        attrs.LIKE_NUM: orth.like_number,
        attrs.LIKE_EMAIL: orth.like_email,
        attrs.IS_STOP: lambda string: False,
        attrs.IS_OOV: lambda string: True
    }
    
    def __init__(self,
        data_dir=None,
        vocab=None,
        tokenizer=None,
        tagger=None,
        parser=None,
        entity=None,
        matcher=None,
        serializer=None,
        vectors=None):
        """
        A model can be specified:

        1) by calling a Language subclass
            - spacy.en.English()

        2) by calling a Language subclass with data_dir
            - spacy.en.English('my/model/root')
            - spacy.en.English(data_dir='my/model/root')

        3) by package name
            - spacy.load('en_default')
            - spacy.load('en_default==1.0.0')

        4) by package name with a relocated package base
            - spacy.load('en_default', via='/my/package/root')
            - spacy.load('en_default==1.0.0', via='/my/package/root')
        """
        if data_dir is None:
            data_dir = util.get_package_by_name(about.__models__[self.lang])

        self.vocab = self.load_vocab(data_dir / 'vocab', given=vocab)
        self.tokenizer = self.load_tokenizer(data_dir / 'tokenizer', given=tokenizer)
        self.tagger = self.load_tagger(data_dir / 'tagger', given=tagger)
        self.entity = self.load_entity(data_dir / 'entity', given=entity)
        self.parser = self.load_parser(data_dir / 'parser', given=parser)
        self.matcher = self.load_matcher(data_dir / 'matcher', given=matcher)

    def load_vocab(self, data_dir, given=None):
        return self.Vocab.load(data_dir) if given in (None, True) else given

    def load_tokenizer(self, data_dir, given=None):
        return self.Tokenizer.load(self.vocab, data_dir) if given in (None, True) else given

    def load_tagger(self, data_dir, given=None):
        return self.Tagger.load(self.vocab, data_dir) if given in (None, True) else given

    def load_parser(self, data_dir, given=None):
        return self.Parser.load(self.vocab, data_dir) if given in (None, True) else given
    
    def load_entity(self, data_dir, given=None):
        return self.Entity.load(self.vocab, data_dir) if given in (None, True) else given
    
    def load_matcher(self, data_dir, given=None):
        return self.Matcher.load(self.vocab, data_dir) if given in (None, True) else given

    def __call__(self, text, tag=True, parse=True, entity=True):
        """Apply the pipeline to some text.  The text can span multiple sentences,
        and can contain arbtrary whitespace.  Alignment into the original string
        is preserved.
        
        Args:
            text (unicode): The text to be processed.

        Returns:
            tokens (spacy.tokens.Doc):

        >>> from spacy.en import English
        >>> nlp = English()
        >>> tokens = nlp('An example sentence. Another example sentence.')
        >>> tokens[0].orth_, tokens[0].head.tag_
        ('An', 'NN')
        """
        tokens = self.tokenizer(text)

        if self.tagger and tag:
            self.tagger(tokens)
        if self.matcher and entity:
            self.matcher(tokens)
        if self.parser and parse:
            self.parser(tokens)
        if self.entity and entity:
            # Add any of the entity labels already set, in case we don't have them.
            for tok in tokens:
                if tok.ent_type != 0:
                    self.entity.add_label(tok.ent_type)
            self.entity(tokens)
        return tokens

    def pipe(self, texts, tag=True, parse=True, entity=True, n_threads=2,
            batch_size=1000):
        stream = self.tokenizer.pipe(texts,
            n_threads=n_threads, batch_size=batch_size)
        if self.tagger and tag:
            stream = self.tagger.pipe(stream,
                n_threads=n_threads, batch_size=batch_size)
        if self.matcher and entity:
            stream = self.matcher.pipe(stream,
                n_threads=n_threads, batch_size=batch_size)
        if self.parser and parse:
            stream = self.parser.pipe(stream,
                n_threads=n_threads, batch_size=batch_size)
        if self.entity and entity:
            stream = self.entity.pipe(stream,
                n_threads=1, batch_size=batch_size)
        for doc in stream:
            yield doc

    def end_training(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        if self.parser:
            self.parser.model.end_training()
            self.parser.model.dump(path.join(data_dir, 'deps', 'model'))
        if self.entity:
            self.entity.model.end_training()
            self.entity.model.dump(path.join(data_dir, 'ner', 'model'))
        if self.tagger:
            self.tagger.model.end_training()
            self.tagger.model.dump(path.join(data_dir, 'pos', 'model'))

        strings_loc = path.join(data_dir, 'vocab', 'strings.json')
        with io.open(strings_loc, 'w', encoding='utf8') as file_:
            self.vocab.strings.dump(file_)
        self.vocab.dump(path.join(data_dir, 'vocab', 'lexemes.bin'))

        if self.tagger:
            tagger_freqs = list(self.tagger.freqs[TAG].items())
        else:
            tagger_freqs = []
        if self.parser:
            dep_freqs = list(self.parser.moves.freqs[DEP].items())
            head_freqs = list(self.parser.moves.freqs[HEAD].items())
        else:
            dep_freqs = []
            head_freqs = []
        if self.entity:
            entity_iob_freqs = list(self.entity.moves.freqs[ENT_IOB].items())
            entity_type_freqs = list(self.entity.moves.freqs[ENT_TYPE].items())
        else:
            entity_iob_freqs = []
            entity_type_freqs = []
        with open(path.join(data_dir, 'vocab', 'serializer.json'), 'w') as file_:
            file_.write(
                json.dumps([
                    (TAG, tagger_freqs),
                    (DEP, dep_freqs),
                    (ENT_IOB, entity_iob_freqs),
                    (ENT_TYPE, entity_type_freqs),
                    (HEAD, head_freqs)
                ]))


#
#    @classmethod
#    def default_vocab(cls, package, get_lex_attr=None, vectors_package=None):
#        if get_lex_attr is None:
#            if package.has_file('vocab', 'oov_prob'):
#                with package.open(('vocab', 'oov_prob')) as file_:
#                    oov_prob = float(file_.read().strip())
#                get_lex_attr = cls.default_lex_attrs(oov_prob=oov_prob)
#            else:
#                get_lex_attr = cls.default_lex_attrs()
#        if hasattr(package, 'dir_path'):
#            return Vocab.from_package(package, get_lex_attr=get_lex_attr,
#                vectors_package=vectors_package)
#        else:
#            return Vocab.load(package, get_lex_attr)
#
#    @classmethod
#    def default_parser(cls, package, vocab):
#        if hasattr(package, 'dir_path'):
#            data_dir = package.dir_path('deps')
#        else:
#            data_dir = package
#        if data_dir and path.exists(data_dir):
#            return Parser.from_dir(data_dir, vocab.strings, ArcEager)
#        else:
#            return None
#
#    @classmethod
#    def default_entity(cls, package, vocab):
#        if hasattr(package, 'dir_path'):
#            data_dir = package.dir_path('ner')
#        else:
#            data_dir = package
#        if data_dir and path.exists(data_dir):
#            return Parser.from_dir(data_dir, vocab.strings, BiluoPushDown)
#        else:
#            return None
#
#
