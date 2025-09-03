
from typing import Iterable
from dataclasses import dataclass

@dataclass
class SimpleTrieNode[K,V]:
    """A node of a Trie.

    This is a reference implementation for the TrieNode class.
    """

    content  : V
    children : dict[K, SimpleTrieNode[K,V]] = dict()

    def __getitem__(self, key:K):
        return self.children[key]

    def __setitem__(self, key:K, node:SimpleTrieNode[K,V]):
        self.children[key] = node

    def __add__(self, other:SimpleTrieNode[K,V]):
        return SimpleTrieNode[K,V](content=self.content,
                                   children=self.children.copy().update(other.children))

@dataclass
class SimpleTrie[K,V]:
    """A Trie.

    This is a reference implementation for the Trie class.
    """
    root: SimpleTrieNode[K,V]

    def __getitem__(self, query:Iterable[K]):
        node = self.root
        results = [node.content]
        for key in query:
            node = node[key]
            results.append(node.content)
        return results

    def __setitem__(self, query:Iterable[K], contents:Iterable[V]):
        node = self.root
        for key, content in zip(query, contents):
            if key in node:
                node = node[key]
            else:
                new = SimpleTrieNode[K,V](content = content)
                node[key] = new
                node = new
        pass

    def __add__(self, other:SimpleTrie[K,V]):
        return SimpleTrie(root=self.root + other.root)






@dataclass
class PrefixTrieNode[K,V]:
    """A node of a Trie that additionally maintains a prefix.

    An edge in a regular Trie typically represents a single character.
    For example,

    * Hello everyone! Hiccups are not great
    * Hello everyone! Hippos are great

    the tree would be

    [root]
      .Hello
        .everyone
          .!
            .Hiccups
              .are
                .not
                  .great
            .Hippos
              .are
                .great

    However, this would cause a large number of recursions
    where most nodes have a single child -- e.g.,
    nodes in "Hello everyone !", "Hiccups are not great", "Hippos are great"
    all have a single child.

    The extension of a prefix allows Trie to avoid such long linear branches.
    While the key to the children is defined in the same manner
    (the first token the substring),
    the prefix maintains the skipped elements.
    The result of the example above would look like

    [root]
    .Hello [everyone !]
      .Hiccups [are not great]
      .Hippos [are great]

    This changes the way merging operation works.
    When two nodes with a different prefix gets merged,
    it creates a new node with the longest common prefix.
    """

    prefix   : Iterable[K]
    content  : Iterable[V]
    children : dict[K, PrefixTrieNode[K,V]] = dict()

    def __post__init__(self):
        assert len(prefix) == len(content)

    def __getitem__(self, key:K):
        return self.children[key]

    def __setitem__(self, key:K, node:PrefixTrieNode[K,V]):
        self.children[key] = node

    def __add__(self, other:PrefixTrieNode[K,V]):

        longest_common_prefix = 0
        for x, y in zip(self.prefix, other.prefix):
            if (x != y):
                break
            longest_common_prefix += 1

        # note: prefix must contain the first token.
        assert longest_common_prefix >= 1, "trying to merge two nodes that have no common prefix"
        return PrefixTrieNode[K,V](
            prefix=self.prefix[:longest_common_prefix],
            content=self.content[:longest_common_prefix],
            children={
                self.prefix[longest_common_prefix] : PrefixTrieNode[K,V](
                    prefix=self.prefix[longest_common_prefix:],
                    content=self.content[longest_common_prefix:],
                    children=self.children)
                other.prefix[longest_common_prefix] : PrefixTrieNode[K,V](
                    prefix=other.prefix[longest_common_prefix:],
                    content=other.content[longest_common_prefix:],
                    children=other.children)
            })

@dataclass
class PrefixTrie[K,V]:
    """A PrefixTrie.

    This is a reference implementation for the PrefixTrie class.
    """
    root: PrefixTrieNode[K,V]

    def __getitem__(self, query:Iterable[K]):
        node = self.root
        results = [node.content]

        index = 0
        while True:
            key = query[index]
            node = node[key]
            prefix_length = len(node.prefix)
            if query[index:index+prefix_length] != node.prefix:
                raise IndexError()
            index += prefix_length
            results.extend(node.content)

        return results

    def __setitem__(self, query:Iterable[K], contents:Iterable[V]):
        node = self.root

        singleton = PrefixTrieNode[K,V](prefix = query,
                                        content = contents)

        self.root = node + singleton
        pass

    def __add__(self, other:PrefixTrie[K,V]):
        return PrefixTrie(root=self.root + other.root)


