#!/usr/bin/env python3
"""
fishbuffet.py
Калькулятор для расчёта рейкбэка Fish Buffet на ПокерОк
"""

import argparse
import math
import random
import statistics
from typing import Dict, List, Any, Optional


# -----------------------------
# Конфигурация уровней Fish Buffet
# -----------------------------

# Все суммы в долларах США (условно).
# Параметры близкие по идее к реальной системе, но НЕ являются точной копией.
FISH_STATUSES: Dict[str, Dict[str, Any]] = {
    "shrimp": {
        "display_name": "Shrimp",
        "points_required": 750,
        "rake_per_point": 0.5,  # $ рейка за 1 point
        "chests": [
            {"reward": 2.0, "prob": 0.50},
            {"reward": 3.0, "prob": 0.30},
            {"reward": 5.0, "prob": 0.15},
            {"reward": 8.0, "prob": 0.05},
        ],
        "expiry_days": 30,
    },
    "goldfish": {
        "display_name": "Goldfish",
        "points_required": 3000,
        "rake_per_point": 0.5,
        "chests": [
            {"reward": 8.0, "prob": 0.45},
            {"reward": 10.0, "prob": 0.30},
            {"reward": 15.0, "prob": 0.20},
            {"reward": 25.0, "prob": 0.05},
        ],
        "expiry_days": 30,
    },
    "crab": {
        "display_name": "Crab",
        "points_required": 9000,
        "rake_per_point": 0.5,
        "chests": [
            {"reward": 25.0, "prob": 0.40},
            {"reward": 35.0, "prob": 0.30},
            {"reward": 50.0, "prob": 0.20},
            {"reward": 80.0, "prob": 0.10},
        ],
        "expiry_days": 30,
    },
    "octopus": {
        "display_name": "Octopus",
        "points_required": 60000,
        "rake_per_point": 0.5,
        "chests": [
            {"reward": 120.0, "prob": 0.40},
            {"reward": 160.0, "prob": 0.30},
            {"reward": 220.0, "prob": 0.20},
            {"reward": 320.0, "prob": 0.10},
        ],
        "expiry_days": 60,
    },
    "whale": {
        "display_name": "Whale",
        "points_required": 300000,
        "rake_per_point": 0.5,
        "chests": [
            {"reward": 650.0, "prob": 0.40},
            {"reward": 900.0, "prob": 0.30},
            {"reward": 1300.0, "prob": 0.20},
            {"reward": 2000.0, "prob": 0.10},
        ],
        "expiry_days": 90,
    },
    "shark": {
        "display_name": "Shark",
        "points_required": 1500000,
        "rake_per_point": 0.5,
        "chests": [
            {"reward": 3500.0, "prob": 0.40},
            {"reward": 5000.0, "prob": 0.30},
            {"reward": 7500.0, "prob": 0.20},
            {"reward": 10000.0, "prob": 0.10},
        ],
        "expiry_days": 90,
    },
}


# -----------------------------
# Исключения
# -----------------------------


class FishBuffetError(Exception):
    """Базовая ошибка для модуля fishbuffet."""


# -----------------------------
# Вспомогательные функции
# -----------------------------


def list_available_statuses() -> str:
    return ", ".join(conf["display_name"] for conf in FISH_STATUSES.values())


def normalize_status_name(name: str) -> str:
    """Нормализует имя статуса (регистронезависимо)."""
    key = name.strip().lower()
    if key in FISH_STATUSES:
        return key

    # Попробуем по display_name
    for k, conf in FISH_STATUSES.items():
        if conf["display_name"].lower() == key:
            return k

    raise FishBuffetError(
        f"Неизвестный статус '{name}'. Доступные статусы: {list_available_statuses()}"
    )


def get_status_config(status_name: str) -> Dict[str, Any]:
    key = normalize_status_name(status_name)
    conf = FISH_STATUSES[key]
    # Проверим сумму вероятностей сундуков
    prob_sum = sum(chest["prob"] for chest in conf["chests"])
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        # Не критично, но предупредим в консоль позже; тут просто оставим.
        pass
    return conf


def calculate_points_from_rake(rake: float, rake_per_point: float) -> float:
    if rake_per_point <= 0:
        raise FishBuffetError("rake_per_point должен быть > 0.")
    return rake / rake_per_point


def calculate_expected_reward(status_conf: Dict[str, Any]) -> float:
    return sum(ch["reward"] * ch["prob"] for ch in status_conf["chests"])


def calculate_effective_rakeback(status_conf: Dict[str, Any]) -> float:
    points_required = status_conf["points_required"]
    rake_per_point = status_conf["rake_per_point"]
    total_rake_for_level = points_required * rake_per_point
    expected_reward = calculate_expected_reward(status_conf)
    if total_rake_for_level <= 0:
        return 0.0
    return expected_reward / total_rake_for_level * 100.0


def calculate_progress(
    status_conf: Dict[str, Any],
    current_points: float,
    daily_rake: Optional[float],
) -> Dict[str, Any]:
    points_required = status_conf["points_required"]
    rake_per_point = status_conf["rake_per_point"]

    current_points = max(current_points, 0.0)
    points_left = max(points_required - current_points, 0.0)
    result: Dict[str, Any] = {
        "points_required": points_required,
        "current_points": current_points,
        "points_left": points_left,
    }

    if daily_rake is not None and daily_rake > 0:
        points_per_day = calculate_points_from_rake(daily_rake, rake_per_point)
        days_to_next = points_left / points_per_day if points_per_day > 0 else math.inf
        result["daily_rake"] = daily_rake
        result["points_per_day"] = points_per_day
        result["days_to_next"] = days_to_next
    else:
        result["daily_rake"] = daily_rake
        result["points_per_day"] = None
        result["days_to_next"] = None

    return result


def calculate_period_stats(
    status_conf: Dict[str, Any],
    daily_rake: Optional[float],
    period_days: Optional[int],
) -> Optional[Dict[str, Any]]:
    if daily_rake is None or daily_rake <= 0 or period_days is None or period_days <= 0:
        return None

    rake_per_point = status_conf["rake_per_point"]
    points_required = status_conf["points_required"]

    total_rake_period = daily_rake * period_days
    total_points_period = calculate_points_from_rake(total_rake_period, rake_per_point)
    expected_chests_period = total_points_period / points_required

    expected_reward = calculate_expected_reward(status_conf)
    expected_rb_money = expected_chests_period * expected_reward
    avg_rb_per_day = expected_rb_money / period_days if period_days > 0 else 0.0

    return {
        "period_days": period_days,
        "total_rake_period": total_rake_period,
        "total_points_period": total_points_period,
        "expected_chests_period": expected_chests_period,
        "expected_rb_money": expected_rb_money,
        "avg_rb_per_day": avg_rb_per_day,
    }


def simulate_chests(
    status_conf: Dict[str, Any],
    num_chests: int,
    iterations: int,
) -> Dict[str, Any]:
    if num_chests <= 0 or iterations <= 0:
        raise FishBuffetError("num_chests и iterations должны быть > 0 для симуляции.")

    rewards = [ch["reward"] for ch in status_conf["chests"]]
    probs = [ch["prob"] for ch in status_conf["chests"]]

    # Накопим суммы наград за итерации
    totals: List[float] = []

    for _ in range(iterations):
        total = 0.0
        for _ in range(num_chests):
            # random.choices позволяет выбирать по весам
            reward = random.choices(rewards, weights=probs, k=1)[0]
            total += reward
        totals.append(total)

    totals.sort()
    mean_val = statistics.fmean(totals)
    min_val = totals[0]
    max_val = totals[-1]
    median_val = statistics.median(totals)
    q25 = totals[int(0.25 * (len(totals) - 1))]
    q75 = totals[int(0.75 * (len(totals) - 1))]

    return {
        "num_chests": num_chests,
        "iterations": iterations,
        "min": min_val,
        "q25": q25,
        "median": median_val,
        "q75": q75,
        "max": max_val,
        "mean": mean_val,
    }


# -----------------------------
# Ввод / вывод
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Расчёт рейкбэка Fish Buffet (PokerOK-подобная система)."
    )
    parser.add_argument(
        "-s",
        "--status",
        help=f"Текущий статус Fish Buffet. Доступные: {list_available_statuses()}",
    )
    parser.add_argument(
        "-p",
        "--points",
        type=float,
        help="Текущее количество Fish Points на уровне.",
    )
    parser.add_argument(
        "-r",
        "--daily-rake",
        type=float,
        help="Средний рейк в день (в $).",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        help="Период анализа в днях (например, 30).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Выполнить Monte Carlo симуляцию наград.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Количество прогонов для симуляции (по умолчанию 10000).",
    )
    parser.add_argument(
        "--chests",
        type=int,
        help=(
            "Количество сундуков для симуляции. "
            "Если не указано, будет вычислено из рейка за период."
        ),
    )
    parser.add_argument(
        "--compare-statuses",
        action="store_true",
        help="Сравнить несколько статусов на одинаковом рейке за период.",
    )
    return parser.parse_args()


def interactive_input() -> Dict[str, Any]:
    print("=== Fish Buffet калькулятор (PokerOK-подобная система) ===")
    print(f"Доступные статусы: {list_available_statuses()}")
    status = input("Введите ваш статус: ").strip()

    points_str = input("Текущее количество Fish Points (можно 0): ").strip()
    points = float(points_str or 0.0)

    daily_rake_str = input("Средний рейк в день, $ (можно 0): ").strip()
    daily_rake = float(daily_rake_str or 0.0)

    period_str = input("Период анализа в днях (Enter, если не нужен): ").strip()
    period_days = int(period_str) if period_str else None

    simulate_answer = input("Запустить Monte Carlo симуляцию? [y/N]: ").strip().lower()
    simulate = simulate_answer == "y"

    iterations = 10000
    if simulate:
        it_str = input("Количество прогонов (по умолчанию 10000): ").strip()
        iterations = int(it_str) if it_str else 10000

    chests = None
    if simulate:
        chests_str = input(
            "Количество сундуков для симуляции (Enter, чтобы вычислить из периода): "
        ).strip()
        chests = int(chests_str) if chests_str else None

    return {
        "status": status,
        "points": points,
        "daily_rake": daily_rake,
        "period_days": period_days,
        "simulate": simulate,
        "iterations": iterations,
        "chests": chests,
        "compare_statuses": False,
    }


def print_status_header(status_key: str, status_conf: Dict[str, Any]) -> None:
    name = status_conf["display_name"]
    print(f"\n=== Статус: {name} ===")
    print(f"Требуемые очки на уровень: {status_conf['points_required']:.0f}")
    print(f"Рейк на 1 point: ${status_conf['rake_per_point']:.2f}")
    total_rake_for_level = status_conf["points_required"] * status_conf["rake_per_point"]
    print(f"Общий рейк на уровень: ${total_rake_for_level:.2f}")

    prob_sum = sum(ch["prob"] for ch in status_conf["chests"])
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        print(
            f"ВНИМАНИЕ: сумма вероятностей сундуков для статуса {name} = {prob_sum:.3f}, "
            f"что отличается от 1.0"
        )


def print_progress(progress: Dict[str, Any]) -> None:
    print("\n--- Прогресс по уровню ---")
    print(f"Текущие очки: {progress['current_points']:.2f}")
    print(f"Осталось до уровня: {progress['points_left']:.2f} points")

    if progress["points_left"] <= 0:
        print("Уровень уже полностью закрыт — поздравляем!")
        return

    daily_rake = progress.get("daily_rake")
    points_per_day = progress.get("points_per_day")
    days_to_next = progress.get("days_to_next")

    if daily_rake is not None and daily_rake > 0 and points_per_day:
        print(f"При рейке ${daily_rake:.2f}/день:")
        print(f"  Очков/день: {points_per_day:.2f}")
        print(f"  Примерно дней до апа: {days_to_next:.2f}")
    else:
        print("Рейк в день не задан или равен 0 — дни до апа посчитать нельзя.")


def print_rakeback_info(status_conf: Dict[str, Any]) -> None:
    print("\n--- Рейкбэк Fish Buffet ---")
    expected_reward = calculate_expected_reward(status_conf)
    rb_percent = calculate_effective_rakeback(status_conf)
    print(f"Ожидаемая награда за сундук: ${expected_reward:.2f}")
    print(f"Эффективный рейкбэк: {rb_percent:.2f}% от рейка на уровень")


def print_period_info(period_stats: Optional[Dict[str, Any]]) -> None:
    if not period_stats:
        print("\n--- Период ---")
        print("Период не задан или рейк в день <= 0 — пропускаем расчёт по периоду.")
        return

    print("\n--- За указанный период ---")
    print(f"Период: {period_stats['period_days']} дней")
    print(f"Рейк за период: ${period_stats['total_rake_period']:.2f}")
    print(f"Fish Points за период: {period_stats['total_points_period']:.2f}")
    print(
        f"Ожидаемое количество сундуков: {period_stats['expected_chests_period']:.2f}"
    )
    print(
        f"Ожидаемый доход Fish Buffet за период: ${period_stats['expected_rb_money']:.2f}"
    )
    print(f"Средний RB/день: ${period_stats['avg_rb_per_day']:.2f}")


def print_simulation_info(sim_stats: Dict[str, Any]) -> None:
    print(
        f"\n--- Результаты Monte Carlo ({sim_stats['iterations']} прогонов, "
        f"{sim_stats['num_chests']} сундуков) ---"
    )
    print(f"Мин: ${sim_stats['min']:.2f}")
    print(f"25% квантиль: ${sim_stats['q25']:.2f}")
    print(f"Медиана: ${sim_stats['median']:.2f}")
    print(f"75% квантиль: ${sim_stats['q75']:.2f}")
    print(f"Макс: ${sim_stats['max']:.2f}")
    print(f"Среднее: ${sim_stats['mean']:.2f}")


def run_compare_statuses(
    daily_rake: Optional[float],
    period_days: Optional[int],
) -> None:
    print("\n=== Сравнение статусов Fish Buffet ===")
    if daily_rake is None or daily_rake <= 0 or period_days is None or period_days <= 0:
        print(
            "Для сравнения статусов необходимо задать --daily-rake > 0 и --period-days > 0."
        )
        return

    header = (
        f"{'Статус':<10}"
        f"{'RB %':>8}"
        f"{'Рейк за период':>18}"
        f"{'Сундуков':>12}"
        f"{'Доход RB':>12}"
        f"{'RB/день':>10}"
    )
    print(header)
    print("-" * len(header))

    for key, conf in FISH_STATUSES.items():
        period_stats = calculate_period_stats(conf, daily_rake, period_days)
        if not period_stats:
            continue
        rb_percent = calculate_effective_rakeback(conf)
        print(
            f"{conf['display_name']:<10}"
            f"{rb_percent:>8.2f}"
            f"{period_stats['total_rake_period']:>18.2f}"
            f"{period_stats['expected_chests_period']:>12.2f}"
            f"{period_stats['expected_rb_money']:>12.2f}"
            f"{period_stats['avg_rb_per_day']:>10.2f}"
        )


# -----------------------------
# main
# -----------------------------


def main() -> None:
    args = parse_args()

    if not any(
        [
            args.status,
            args.points is not None,
            args.daily_rake is not None,
            args.period_days is not None,
            args.simulate,
            args.compare_statuses,
        ]
    ):
        # Ничего не передано — интерактивный режим
        data = interactive_input()
        status_name = data["status"]
        points = data["points"]
        daily_rake = data["daily_rake"]
        period_days = data["period_days"]
        simulate = data["simulate"]
        iterations = data["iterations"]
        num_chests_for_sim = data["chests"]
        compare_statuses = data["compare_statuses"]
    else:
        # Режим через аргументы
        status_name = args.status
        points = args.points if args.points is not None else 0.0
        daily_rake = args.daily_rake
        period_days = args.period_days
        simulate = args.simulate
        iterations = args.iterations
        num_chests_for_sim = args.chests
        compare_statuses = args.compare_statuses

    if compare_statuses:
        run_compare_statuses(daily_rake, period_days)
        # Если статус не задан, можно завершиться
        if not status_name:
            return
        print("\nДополнительно расчитаем детали для выбранного статуса.\n")

    if not status_name:
        # В аргументах не задан статус, запросим в интерактиве
        status_name = input(
            f"Статус не указан. Введите статус ({list_available_statuses()}): "
        ).strip()

    try:
        status_conf = get_status_config(status_name)
    except FishBuffetError as e:
        print(f"Ошибка: {e}")
        return

    print_status_header(normalize_status_name(status_name), status_conf)

    progress = calculate_progress(status_conf, points, daily_rake)
    print_progress(progress)

    print_rakeback_info(status_conf)

    period_stats = calculate_period_stats(status_conf, daily_rake, period_days)
    print_period_info(period_stats)

    # Симуляция
    if simulate:
        # Определим число сундуков
        if num_chests_for_sim is None:
            if period_stats:
                # Округлим ожидаемое количество до ближайшего целого, но минимум 1
                est_chests = period_stats["expected_chests_period"]
                num_chests_for_sim = max(int(round(est_chests)), 1)
            else:
                # Нет периода — спросим у пользователя
                try:
                    ch_str = input(
                        "Период не задан, введите вручную количество сундуков для симуляции: "
                    ).strip()
                    num_chests_for_sim = int(ch_str)
                except Exception:
                    print("Неверный ввод количества сундуков — симуляция отменена.")
                    num_chests_for_sim = None

        if num_chests_for_sim is not None and num_chests_for_sim > 0:
            sim_stats = simulate_chests(status_conf, num_chests_for_sim, iterations)
            print_simulation_info(sim_stats)
        else:
            print("Симуляция не выполнена: количество сундуков не задано или некорректно.")


if __name__ == "__main__":
    main()