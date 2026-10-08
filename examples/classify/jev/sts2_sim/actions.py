"""Action id vocabulary, aligned with STS2MCP's singleplayer action names.

Every candidate id the RunLoop emits is `"<mcp_action>[:<arg>[:<arg>...]]"`
where `<mcp_action>` is the `"action"` field STS2MCP's
`POST /api/v1/singleplayer` accepts (see Gennadiyev/STS2MCP
`McpMod.Actions.cs`). Arguments are sim-native (card ids, enemy slots,
option indices); a live adapter translates them into STS2MCP's request
params (e.g. card_id -> hand `card_index`, enemy slot -> target entity id).

Sharing the action name keeps the sim and the live-game adapter on one id
space, so a policy trained against the sim scores the same candidate ids
it will see when Jev requests are forwarded through the MCP.
"""

from __future__ import annotations

from typing import Final


# --- STS2MCP singleplayer actions -----------------------------------------
# Implemented by the sim today are marked (*). The rest are reserved so later
# phases reuse the exact MCP name instead of inventing a new one.

MENU_SELECT: Final = "menu_select"                          # (*) game_over only
PLAY_CARD: Final = "play_card"                              # (*)
USE_POTION: Final = "use_potion"                            # (*)
DISCARD_POTION: Final = "discard_potion"                    # (*)
END_TURN: Final = "end_turn"                                # (*)
CHOOSE_MAP_NODE: Final = "choose_map_node"                  # (*)
CHOOSE_EVENT_OPTION: Final = "choose_event_option"          # (*)
ADVANCE_DIALOGUE: Final = "advance_dialogue"
CHOOSE_REST_OPTION: Final = "choose_rest_option"            # (*)
SHOP_PURCHASE: Final = "shop_purchase"                      # (*)
CLAIM_REWARD: Final = "claim_reward"                        # (*)
SELECT_CARD_REWARD: Final = "select_card_reward"            # (*)
SKIP_CARD_REWARD: Final = "skip_card_reward"                # (*)
PROCEED: Final = "proceed"                                  # (*)
SELECT_CARD: Final = "select_card"                          # (*)
CONFIRM_SELECTION: Final = "confirm_selection"              # (*)
CANCEL_SELECTION: Final = "cancel_selection"                # (*)
SELECT_BUNDLE: Final = "select_bundle"
CONFIRM_BUNDLE_SELECTION: Final = "confirm_bundle_selection"
CANCEL_BUNDLE_SELECTION: Final = "cancel_bundle_selection"
COMBAT_SELECT_CARD: Final = "combat_select_card"            # (*)
COMBAT_CONFIRM_SELECTION: Final = "combat_confirm_selection"  # (*)
SELECT_RELIC: Final = "select_relic"
SKIP_RELIC_SELECTION: Final = "skip_relic_selection"
CLAIM_TREASURE_RELIC: Final = "claim_treasure_relic"        # (*)
CRYSTAL_SPHERE_SET_TOOL: Final = "crystal_sphere_set_tool"
CRYSTAL_SPHERE_CLICK_CELL: Final = "crystal_sphere_click_cell"
CRYSTAL_SPHERE_PROCEED: Final = "crystal_sphere_proceed"

MCP_ACTIONS: Final = frozenset({
    MENU_SELECT, PLAY_CARD, USE_POTION, DISCARD_POTION, END_TURN,
    CHOOSE_MAP_NODE, CHOOSE_EVENT_OPTION, ADVANCE_DIALOGUE, CHOOSE_REST_OPTION,
    SHOP_PURCHASE, CLAIM_REWARD, SELECT_CARD_REWARD, SKIP_CARD_REWARD, PROCEED,
    SELECT_CARD, CONFIRM_SELECTION, CANCEL_SELECTION, SELECT_BUNDLE,
    CONFIRM_BUNDLE_SELECTION, CANCEL_BUNDLE_SELECTION, COMBAT_SELECT_CARD,
    COMBAT_CONFIRM_SELECTION, SELECT_RELIC, SKIP_RELIC_SELECTION,
    CLAIM_TREASURE_RELIC, CRYSTAL_SPHERE_SET_TOOL, CRYSTAL_SPHERE_CLICK_CELL,
    CRYSTAL_SPHERE_PROCEED,
})

# Fixed ids.
NEOW_SKIP: Final = f"{CHOOSE_EVENT_OPTION}:0"
GAME_OVER_MAIN_MENU: Final = f"{MENU_SELECT}:main_menu"

_SEP: Final = ":"


def format_action(name: str, *args: object) -> str:
    """Build an action id. `name` must be an STS2MCP action name."""

    if name not in MCP_ACTIONS:
        raise ValueError(f"unknown STS2MCP action {name!r}")
    parts = [name, *(str(a) for a in args)]
    for part in parts[1:]:
        if not part or _SEP in part:
            raise ValueError(f"invalid action arg {part!r} for {name!r}")
    return _SEP.join(parts)


def parse_action(action_id: str) -> tuple[str, tuple[str, ...]]:
    """Split an action id into `(mcp_action_name, args)`."""

    name, *args = action_id.split(_SEP)
    if name not in MCP_ACTIONS:
        raise ValueError(f"unknown STS2MCP action in id {action_id!r}")
    if any(not a for a in args):
        raise ValueError(f"empty arg in action id {action_id!r}")
    return name, tuple(args)
