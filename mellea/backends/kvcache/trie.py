
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

    def items(self):
        """
        Walk over the keys and the values stored in the radix tree
        """
        if len(self.children) == 0:  # leaf node
            yield self.prefix, self.content
        else:
            for _, node in sorted(self.children.items()):
                for prefix, content in node.items():
                    yield (self.prefix + prefix), (self.content + content)

    def keys(self):
        """
        Walk over the keys stored in the radix tree
        """
        if len(self.children) == 0:  # leaf node
            yield self.prefix
        else:
            for _, node in sorted(self.children.items()):
                for prefix in node.keys():
                    yield self.prefix + prefix

    def values(self):
        """
        Walk over the values stored in the radix tree
        """
        if len(self.children) == 0:  # leaf node
            yield self.content
        else:
            for _, node in sorted(self.children.items()):
                for content in node.values():
                    yield self.content + content

    def nodes(self):
        """
        Walk over the nodes in the radix tree
        """
        yield self
        for _, node in sorted(self.children.items()):
            yield from node.nodes()

    def branching_prefixes(self):
        """
        Walk over every prefix that is followed by a branching point, and the list of choices
        """
        if self.children:
            yield self.prefix, sorted(self.children.keys())
        for _, node in sorted(self.children.items()):
            for prefix, choices in node.branching_prefixes():
                yield self.prefix + prefix, choices

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

    def __contains__(self, query:list[K]) -> bool:
        node = self.root

        k_index = 0             # key index
        p_index = 0             # prefix index
        for k_index, key in enumerate(query):
            if p_index > len(node.prefix):
                raise RuntimeError("huh?")
            elif p_index == len(node.prefix):
                if key in node.children:
                    node = node.children[key]
                else:
                    return False
                p_index = 0
            else:
                pass

            if node.prefix[p_index] == key:
                p_index += 1
            else:
                return False

        return True

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

    def items(self):
        yield from self.root.items()

    def keys(self):
        yield from self.root.keys()

    def values(self):
        yield from self.root.values()

    def nodes(self):
        yield from self.root.nodes()

    def branching_prefixes(self):
        yield from self.root.branching_prefixes()

    @staticmethod
    def from_map(data:list[tuple[list[K],list[V]]]):
        t = RadixTrie[K,V]()
        for ks, vs in data:
            t[ks] = vs
        return t


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
    for keys, values in (n2+n3).items():
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

    print("testing key-value iterator")
    for keys, values in t.items():
        print(keys, values)

    print("testing key iterator")
    for keys in t.keys():
        print(keys)

    print("testing value iterator")
    for values in t.values():
        print(values)

    print("testing node iterator")
    for n in t.nodes():
        print(n)

    print("testing branching point iterator")
    for prefix, choices in t.branching_prefixes():
        print(prefix, choices)

