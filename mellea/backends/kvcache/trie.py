
from __future__ import annotations

from typing import Iterable, TypeVar
from dataclasses import dataclass, field

K = TypeVar("K")
V = TypeVar("V")

@dataclass
class RadixTrieNode[K,V]:
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

    prefix   : list[K] = field(default_factory=list)
    content  : list[V] = field(default_factory=list)
    children : dict[K, RadixTrieNode[K,V]] = field(default_factory=dict)

    def __post__init__(self):
        assert len(self.prefix) == len(self.content)

    def __contains__(self, key:K) -> bool:
        return key in self.children

    def __getitem__(self, key:K) -> RadixTrieNode[K,V]:
        return self.children[key]

    def __setitem__(self, key:K, node:RadixTrieNode[K,V]):
        self.children[key] = node

    def __add__(self, other:RadixTrieNode[K,V]) -> RadixTrieNode[K,V]:

        longest_common_prefix = 0
        for x, y in zip(self.prefix, other.prefix):
            if (x != y):
                break
            longest_common_prefix += 1

        # note: prefix must contain the first token.
        assert longest_common_prefix >= 1, "trying to merge two nodes that have no common prefix"

        if len(self.prefix) == longest_common_prefix and \
           len(other.prefix) == longest_common_prefix:
            # same length
            k1 = set(self.children.keys())
            k2 = set(other.children.keys())
            k12 = k1 & k2
            k1 = k1 - k12       # only in self
            k2 = k2 - k12       # only in other
            return RadixTrieNode[K,V](
                self.prefix, self.content,
                {
                    **{ k : self.children[k] for k in k1},
                    **{ k : other.children[k] for k in k2},
                    **{ k : self.children[k] + other.children[k] for k in k12},
                })

        if len(self.prefix) == longest_common_prefix and \
           len(other.prefix) > longest_common_prefix:
            # self is subsumed in other
            return self + RadixTrieNode[K,V](
                prefix=other.prefix[:longest_common_prefix],
                content=other.content[:longest_common_prefix],
                children={
                    other.prefix[longest_common_prefix] : RadixTrieNode[K,V](
                        prefix=other.prefix[longest_common_prefix:],
                        content=other.content[longest_common_prefix:],
                        children=other.children),
                })

        if len(self.prefix) > longest_common_prefix and \
           len(other.prefix) == longest_common_prefix:
            # other is subsumed in self
            return other + RadixTrieNode[K,V](
                prefix=self.prefix[:longest_common_prefix],
                content=self.content[:longest_common_prefix],
                children={
                    self.prefix[longest_common_prefix] : RadixTrieNode[K,V](
                        prefix=self.prefix[longest_common_prefix:],
                        content=self.content[longest_common_prefix:],
                        children=self.children),
                })

        return RadixTrieNode[K,V](
            prefix=self.prefix[:longest_common_prefix],
            content=self.content[:longest_common_prefix],
            children={
                self.prefix[longest_common_prefix] : RadixTrieNode[K,V](
                    prefix=self.prefix[longest_common_prefix:],
                    content=self.content[longest_common_prefix:],
                    children=self.children),
                other.prefix[longest_common_prefix] : RadixTrieNode[K,V](
                    prefix=other.prefix[longest_common_prefix:],
                    content=other.content[longest_common_prefix:],
                    children=other.children),
            })

    def __iter__(self):
        if len(self.children) == 0:  # leaf node
            yield self.prefix, self.content
        else:
            for node in self.children.values():
                for prefix, content in node:
                    yield (self.prefix + prefix), (self.content + content)


    def visualize(self, indent=0) -> str:
        s = " "*indent + "prefix: " + str(self.prefix) + "\n"
        for key, value in self.children.items():
            s += " "*indent + "next: " + str(key) + " ->\n"
            s += value.visualize(indent+2)
        return s


@dataclass
class RadixTrie[K,V]:
    """A RadixTrie.

    This is a reference implementation for the RadixTrie class.
    """
    root: RadixTrieNode[K,V] | None = None

    def __getitem__(self, query:list[K]) -> list[V]:
        node = self.root
        stack = []

        k_index = 0             # key index
        p_index = 0             # prefix index
        for k_index, key in enumerate(query):
            if p_index > len(node.prefix):
                raise RuntimeError("huh?")
            elif p_index == len(node.prefix):
                if key in node.children:
                    node = node.children[key]
                else:
                    raise KeyError()
                    # return RadixTrieNode(prefix=query,
                    #                      content=stack)
                p_index = 0
            else:
                pass

            if node.prefix[p_index] == key:
                stack.append(node.content[p_index])
                p_index += 1
            else:
                raise KeyError()

        return stack

    def __setitem__(self, query:list[K], contents:list[V]):

        singleton = RadixTrieNode[K,V](prefix = query,
                                       content = contents)
        if self.root is None:
            self.root = singleton
        else:
            self.root += singleton
        pass

    def __add__(self, other:RadixTrie[K,V]) -> RadixTrie[K,V]:
        return RadixTrie(root=self.root + other.root)

    def visualize(self) -> str:
        return self.root.visualize()

    def __iter__(self):
        yield from iter(self.root)


if __name__ == "__main__":

    n1 = RadixTrieNode[str,int](
        prefix="abc",
        content=[1,2,3])
    print(n1)
    n2 = RadixTrieNode[str,int](
        prefix="abcdef",
        content=[1,2,3,4,5,6])
    print(n2)
    n3 = RadixTrieNode[str,int](
        prefix="abcghi",
        content=[1,2,3,7,8,9])
    print(n3)

    print("n1+n2")
    print(n1+n2)
    print(n1)
    print(n2)

    print("n2+n3")
    print(n2+n3)
    print(n2)
    print(n3)

    print("testing iterator")
    for keys, values in n2+n3:
        print(keys, values)

    print("making radixtrie")
    t = RadixTrie[str,int]()
    t["abc"] = list(range(3))
    print(t)
    t["abcdef"] = list(range(6))
    print(t)
    t["abcghi"] = list(range(6))
    print(t)

    print("testing partial retrieval")
    print(t["a"])
    print(t["ab"])
    print(t["abc"])
    print(t["abcd"])
    print(t["abcde"])
    print(t["abcdef"])

    print("testing iterator")
    for keys, values in t:
        print(keys, values)
