"""Split a parsed receipt among people.

The arithmetic looks trivial and is not. Three things make it awkward:

1. Shared items rarely divide evenly. A $10 appetizer split three ways is
   $3.333..., and money has two decimal places.
2. Tax and tip belong to everyone, proportionally to what they ordered --
   not split evenly, or the person who had a salad subsidizes the person
   who had a steak.
3. Both of the above round, and rounded parts do not have to sum back to
   the total. Handing out shares that add up to a cent more or less than
   the bill is the single most visible bug this code could have.

The fix for (3) is the largest remainder method: compute every share
exactly, floor each to cents, then hand the leftover pennies to whoever
was rounded down hardest. Shares always sum to the total, and the penny
goes to whoever has the strongest claim to it.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from pydantic import BaseModel, Field

from receipt_parser.models import ParsedReceipt

CENT = Decimal("0.01")


class Assignment(BaseModel):
    """Who is paying for one line item.

    More than one name means the item is shared and split evenly among
    them.
    """

    item_index: int
    people: list[str] = Field(min_length=1)


class PersonShare(BaseModel):
    name: str
    items_subtotal: Decimal
    tax: Decimal
    tip: Decimal
    total: Decimal


class SplitResult(BaseModel):
    shares: list[PersonShare]
    unassigned_item_indices: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return sum((s.total for s in self.shares), Decimal("0"))


def split_bill(
    receipt: ParsedReceipt,
    assignments: list[Assignment],
) -> SplitResult:
    """Divide a receipt among people.

    Unassigned items are reported rather than silently distributed. A
    forgotten item means someone is about to be undercharged, and quietly
    spreading it across everyone hides that.
    """
    warnings: list[str] = []

    people = _people_in_order(assignments)
    if not people:
        return SplitResult(shares=[], warnings=["No one was assigned any items."])

    assigned_indices = {a.item_index for a in assignments}
    unassigned = [
        i for i in range(len(receipt.line_items)) if i not in assigned_indices
    ]
    if unassigned:
        names = ", ".join(
            receipt.line_items[i].description for i in unassigned[:3]
        )
        warnings.append(
            f"{len(unassigned)} item(s) not assigned to anyone: {names}. "
            "These are excluded from the split."
        )

    subtotals = _item_subtotals(receipt, assignments, people)
    assigned_sum = sum(subtotals.values(), Decimal("0"))

    tax_shares = _allocate(receipt.tax or Decimal("0"), subtotals, people, assigned_sum)
    tip_shares = _allocate(receipt.tip or Decimal("0"), subtotals, people, assigned_sum)

    shares = [
        PersonShare(
            name=name,
            items_subtotal=subtotals[name],
            tax=tax_shares[name],
            tip=tip_shares[name],
            total=subtotals[name] + tax_shares[name] + tip_shares[name],
        )
        for name in people
    ]

    if receipt.status.value != "ok":
        warnings.append(
            "The receipt did not reconcile, so these amounts may be wrong. "
            "Check the parsed items before paying."
        )

    return SplitResult(
        shares=shares,
        unassigned_item_indices=unassigned,
        warnings=warnings,
    )


def _people_in_order(assignments: list[Assignment]) -> list[str]:
    """Distinct names, in the order they first appear.

    Order is stable so that the leftover-penny rule is deterministic --
    the same input always produces the same split, which matters when two
    people compare their screens.
    """
    seen: list[str] = []
    for assignment in assignments:
        for name in assignment.people:
            if name not in seen:
                seen.append(name)
    return seen


def _item_subtotals(
    receipt: ParsedReceipt,
    assignments: list[Assignment],
    people: list[str],
) -> dict[str, Decimal]:
    """What each person owes for items, before tax and tip."""
    totals = {name: Decimal("0") for name in people}

    for assignment in assignments:
        if not 0 <= assignment.item_index < len(receipt.line_items):
            continue
        item = receipt.line_items[assignment.item_index]
        for name, amount in zip(
            assignment.people,
            _divide(item.total, len(assignment.people)),
        ):
            totals[name] += amount

    return totals


def _divide(amount: Decimal, parts: int) -> list[Decimal]:
    """Split an amount into `parts` shares that sum exactly to it.

    A $10.00 item split three ways becomes 3.34, 3.33, 3.33 -- not three
    shares of 3.33 that lose a penny, and not three of 3.34 that invent
    one.
    """
    if parts <= 0:
        return []

    base = (amount / parts).quantize(CENT, rounding=ROUND_DOWN)
    shares = [base] * parts

    leftover = amount - base * parts
    pennies = int((leftover / CENT).to_integral_value())
    for i in range(pennies):
        shares[i % parts] += CENT

    return shares


def _allocate(
    amount: Decimal,
    weights: dict[str, Decimal],
    people: list[str],
    weight_sum: Decimal,
) -> dict[str, Decimal]:
    """Distribute `amount` in proportion to `weights`, summing exactly.

    Largest remainder: floor everyone's exact share to cents, then give
    the remaining pennies to the people whose exact share was cut by the
    most. Ties break on the stable person order.
    """
    if amount == 0 or weight_sum == 0:
        return {name: Decimal("0") for name in people}

    exact = {name: amount * weights[name] / weight_sum for name in people}
    floored = {
        name: value.quantize(CENT, rounding=ROUND_DOWN)
        for name, value in exact.items()
    }

    distributed = sum(floored.values(), Decimal("0"))
    pennies = int(((amount - distributed) / CENT).to_integral_value())

    if pennies > 0:
        by_remainder = sorted(
            people,
            key=lambda n: (-(exact[n] - floored[n]), people.index(n)),
        )
        for name in by_remainder[:pennies]:
            floored[name] += CENT

    return floored
