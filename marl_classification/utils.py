from argparse import ArgumentParser, Namespace, Action
from collections import Counter
from typing import Any, Union, Sequence, Text, Optional


class SetAppendAction(Action):
    def __call__(
            self, parser: ArgumentParser, namespace: Namespace,
            values: Union[Text, Sequence[Any], None],
            option_string: Optional[Text] = ...
    ) -> None:
        unique_values = set(values)

        if len(unique_values) != len(values):
            dupl_values = [
                item
                for item, count in Counter(values).items()
                if count > 1
            ]

            error_msg = f"duplicates value(s) found for " \
                        f"\"{self.option_strings[-1]}\": " \
                        f"{dupl_values}"

            parser.error(error_msg)
            exit(1)

        setattr(namespace, self.dest, list(unique_values))
