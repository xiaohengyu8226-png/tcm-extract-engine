"""tcm-extract-engine CLI：自动化抽取(C/A/B) + 自反馈自优化(GEPA 飞轮)。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from tcm_extract_engine.common.log import setup_logging

if TYPE_CHECKING:
    from tcm_extract_engine.runner import BookPipeline

# 所有 CLI 子命令共享统一日志配置
setup_logging()

app = typer.Typer(
    name="tcm-engine",
    help="中医自动化抽取与自反馈自优化引擎（C/A/B + GEPA 飞轮）",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
book_app = typer.Typer(
    name="book",
    help="Artifact-Driven 书籍流水线：run/resume/rerun/retry",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(book_app, name="book")
gepa_app = typer.Typer(
    name="gepa",
    help="GEPA 提示词进化：evolve（跑进化）/ promote（人工晋升）/ runs（列产物）",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(gepa_app, name="gepa")
console = Console()


# ── Web 服务（逐步控制抽取前端）──────────────────────────────────────────


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option("--host", help="监听地址")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="监听端口")] = 8692,
    reload: Annotated[bool, typer.Option("--reload", help="开发模式热重载")] = False,
) -> None:
    """启动 Web 前端：逐步控制抽取流水线（驱动 BookPipeline）。"""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]缺少 Web 依赖:[/] 请先 [bold]uv sync --extra web[/]")
        raise typer.Exit(1)
    console.print(f"[bold]tcm-engine Web[/] http://{host}:{port}  (逐步控制抽取)")
    uvicorn.run("tcm_extract_engine.web.app:app", host=host, port=port, reload=reload)


# ── 书籍流水线子命令（Artifact-Driven）────────────────────────────────────


BooksRoot = Annotated[
    Path | None,
    typer.Option("--books-root", help="书籍工作区根目录（默认 workspace/books）"),
]


def _make_pipeline(books_root: Path | None) -> BookPipeline:
    from tcm_extract_engine.runner import BookPipeline

    return BookPipeline(books_root)


async def _run_book_command(
    books_root: BooksRoot,
    coro_factory: Callable[[BookPipeline], Awaitable[Any]],
    *,
    success_verb: str,
) -> Any:
    from tcm_extract_engine.core.manifest import BookManifest

    pipeline = _make_pipeline(books_root)
    try:
        manifest: BookManifest = await coro_factory(pipeline)
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        await pipeline.close()
    console.print(
        f"[green]✓[/] {success_verb}完成: {manifest.book_id} 状态={manifest.status.value}"
    )
    for stage, record in manifest.stages.items():
        console.print(
            f"  [dim]{stage:<10} {record.status:<8} "
            f"records={record.records if record.records is not None else '-'}[/]"
        )
    return manifest


@book_app.command("run")
def book_run(
    source: Annotated[Path, typer.Argument(help="源文件或目录（PDF/Markdown）")],
    books_root: BooksRoot = None,
    domain: Annotated[str, typer.Option("--domain", help="领域包：general/tcm")] = "general",
    confirm_plan: Annotated[
        bool,
        typer.Option(
            "--confirm-plan/--no-confirm-plan",
            help="自动确认抽取计划（默认；--no-confirm-plan 停在人工确认门",
        ),
    ] = True,
) -> None:
    """批量建书并跑到核验完成（止于 WAITING_REVIEW）。"""
    if not source.exists():
        console.print(f"[red]错误:[/] 路径不存在: {source}")
        raise typer.Exit(1)

    files = (
        sorted(
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".markdown"}
        )
        if source.is_dir()
        else [source]
    )
    if not files:
        console.print(f"[red]错误:[/] 未找到可处理文件: {source}")
        raise typer.Exit(1)

    console.print(f"[bold]共 {len(files)} 个源文件待处理[/]\n")
    failed: list[str] = []
    for path in files:
        console.print(f"[bold]处理:[/] {path.name}")
        try:
            asyncio.run(
                _run_book_command(
                    books_root,
                    lambda p, path=path: p.run(path, confirm_plan=confirm_plan, domain_id=domain),
                    success_verb="运行",
                )
            )
        except typer.Exit:
            failed.append(path.name)
    if failed:
        console.print(f"\n[red]失败 {len(failed)} 个:[/] {', '.join(failed)}")
        raise typer.Exit(1)


@book_app.command("run-book")
def book_run_one(
    source: Annotated[Path, typer.Argument(help="单个源文件（PDF/Markdown）")],
    books_root: BooksRoot = None,
    domain: Annotated[str, typer.Option("--domain", help="领域包：general/tcm")] = "general",
    confirm_plan: Annotated[bool, typer.Option("--confirm-plan/--no-confirm-plan")] = True,
) -> None:
    """单本书跑到核验完成。"""
    asyncio.run(
        _run_book_command(
            books_root,
            lambda p: p.run(source, confirm_plan=confirm_plan, domain_id=domain),
            success_verb="运行",
        )
    )


@book_app.command("resume")
def book_resume(
    book_id: Annotated[str, typer.Argument(help="书籍 ID（12 位 hash）")],
    books_root: BooksRoot = None,
) -> None:
    """继续未完成的书（跳过已完成阶段）。"""
    asyncio.run(
        _run_book_command(
            books_root,
            lambda p: p.resume(book_id),
            success_verb="续跑",
        )
    )


@book_app.command("rerun")
def book_rerun(
    book_id: Annotated[str, typer.Argument(help="书籍 ID")],
    from_stage: Annotated[
        str,
        typer.Option(
            "--from", help="从该阶段强制重跑（parse/profile/plan/extract/validate/verify）"
        ),
    ],
    books_root: BooksRoot = None,
) -> None:
    """从指定阶段强制重跑（下游阶段作废）。"""
    asyncio.run(
        _run_book_command(
            books_root,
            lambda p: p.rerun(book_id, from_stage=from_stage),
            success_verb="重跑",
        )
    )


@book_app.command("retry")
def book_retry(
    book_id: Annotated[str, typer.Argument(help="书籍 ID")],
    books_root: BooksRoot = None,
) -> None:
    """重置 FAILED 后继续。"""
    asyncio.run(
        _run_book_command(
            books_root,
            lambda p: p.retry(book_id),
            success_verb="重试",
        )
    )


@book_app.command("confirm")
def book_confirm(
    book_id: Annotated[str, typer.Argument(help="书籍 ID")],
    books_root: BooksRoot = None,
) -> None:
    """人工确认抽取计划（WAITING_PLAN_CONFIRMATION → PLAN_CONFIRMED）。"""
    pipeline = _make_pipeline(books_root)
    try:
        manifest = pipeline.confirm_plan(book_id)
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]✓[/] 计划已确认: {manifest.book_id}")


@book_app.command("status")
def book_status_cmd(
    book_id: Annotated[str, typer.Argument(help="书籍 ID")],
    books_root: BooksRoot = None,
) -> None:
    """查看单本书的 manifest 状态。"""
    pipeline = _make_pipeline(books_root)
    try:
        manifest = pipeline.status(book_id)
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title=f"书籍 {manifest.book_id}")
    table.add_column("字段", style="cyan")
    table.add_column("值", style="green")
    table.add_row("文件", manifest.file_name)
    table.add_row("状态", manifest.status.value)
    table.add_row("hash", manifest.file_hash[:23] + "…")
    if manifest.error:
        table.add_row("[red]错误[/]", manifest.error[:120])
    for stage in ("parse", "profile", "plan", "extract", "validate", "verify", "review", "publish"):
        record = manifest.stages.get(stage)
        if record:
            table.add_row(
                stage,
                f"{record.status} (records={record.records})"
                + (f" err={record.error[:60]}" if record.error else ""),
            )
    console.print(table)


@book_app.command("list")
def book_list(books_root: BooksRoot = None) -> None:
    """列出全部书籍及状态。"""
    pipeline = _make_pipeline(books_root)
    manifests = pipeline.list_all()
    if not manifests:
        console.print("[dim]暂无书籍工作目录[/]")
        return

    table = Table(title="书籍流水线")
    table.add_column("Book ID", style="cyan")
    table.add_column("文件")
    table.add_column("状态", style="green")
    table.add_column("记录数", justify="right")
    for m in manifests:
        extract = m.stages.get("extract")
        table.add_row(
            m.book_id,
            m.file_name[:30],
            m.status.value,
            str(extract.records) if extract and extract.records else "-",
        )
    console.print(table)


# ── GEPA 进化子命令 ──────────────────────────────────────────────────────────


def _gepa_roots(books_root: Path | None, out_root: Path | None) -> tuple[Path, Path]:
    from tcm_extract_engine.common.paths import project_paths

    br = books_root or project_paths.books
    oroot = out_root or project_paths.gepa
    return br, oroot


@gepa_app.command("evolve")
def gepa_evolve(
    book_id: Annotated[str, typer.Argument(help="书籍 ID（12 位 hash）")],
    template: Annotated[
        str, typer.Option("--template", "-t", help="被进化的 A 模板文件名")
    ] = "general_rule.md",
    books_root: BooksRoot = None,
    out_root: Annotated[
        Path | None, typer.Option("--out-root", help="进化产物根目录（默认 workspace/gepa）")
    ] = None,
    section_id: Annotated[
        list[str] | None,
        typer.Option("--section", help="只进化指定章节（可多次；默认全部同模板章节）"),
    ] = None,
    population: Annotated[int, typer.Option("--population", help="种群大小")] = 2,
    generations: Annotated[int, typer.Option("--generations", help="最大代数")] = 2,
    budget: Annotated[int, typer.Option("--budget", help="评估调用预算上限")] = 4,
    verify_sample: Annotated[
        int, typer.Option("--verify-sample", help="B 核验采样上限；-1=全量")
    ] = 20,
    selection_strategy: Annotated[
        str, typer.Option("--selection", help="父代选择：pareto|scalar")
    ] = "pareto",
    no_baseline: Annotated[bool, typer.Option("--no-baseline", help="跳过基线评估")] = False,
) -> None:
    """对「一本书 × 一个 A 模板」跑小预算 GEPA 进化并落盘产物（不碰 prompts/）。"""
    from tcm_extract_engine.gepa.runner import evolve_template

    br, oroot = _gepa_roots(books_root, out_root)

    def on_gen(rep: Any) -> None:
        console.print(
            f"  [dim]gen {rep.generation}[/] best={rep.best_fitness:.4f} avg={rep.avg_fitness:.4f}"
        )

    async def _run() -> Any:
        return await evolve_template(
            book_id=book_id,
            template=template,
            books_root=br,
            out_root=oroot,
            section_ids=list(section_id) if section_id else None,
            population=population,
            generations=generations,
            budget=budget,
            verify_sample=None if verify_sample < 0 else verify_sample,
            selection_strategy=selection_strategy,
            with_baseline=not no_baseline,
            on_generation=on_gen,
        )

    console.print(f"[bold]GEPA 进化[/] book={book_id} template={template}\n")
    try:
        result = asyncio.run(_run())
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title=f"进化结果 {result.run_id}")
    table.add_column("字段", style="cyan")
    table.add_column("值", style="green")
    base = f"{result.baseline_fitness:.4f}" if result.baseline_fitness is not None else "-"
    table.add_row("baseline fitness", base)
    table.add_row("best fitness", f"{result.best_fitness:.4f}")
    table.add_row("是否提升", "✓" if result.improved else "✗")
    table.add_row("轨迹", str([round(f, 4) for f in result.fitness_history]))
    table.add_row("评估次数", str(result.eval_calls))
    table.add_row("代数", str(result.total_generations))
    table.add_row("胜出变异", result.best_mutation[:60] or "-")
    table.add_row("产物目录", str(result.run_dir))
    console.print(table)
    console.print("\n[bold]胜出约束块：[/]")
    console.print(result.best_constraint or "[dim]（空，等同基线）[/]")
    console.print(f"\n[yellow]晋升：[/] tcm-engine gepa promote {result.run_dir} --reviewer <you>")


@gepa_app.command("promote")
def gepa_promote(
    run_dir: Annotated[Path, typer.Argument(help="evolve 产物目录（含 run.json）")],
    template: Annotated[
        str | None, typer.Option("--template", "-t", help="覆盖 run 内的模板名")
    ] = None,
    reviewer: Annotated[str, typer.Option("--reviewer", "-r", help="审核人标识")] = "cli-user",
    prompts_root: Annotated[
        Path | None, typer.Option("--prompts-root", help="提示词根目录（默认 prompts/）")
    ] = None,
    out_root: Annotated[
        Path | None, typer.Option("--out-root", help="进化产物根目录（用于写趋势台账）")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只预览版本变化，不落盘")] = False,
) -> None:
    """人工门：把胜出约束块写入 prompts/a/<template> 的受管标记块（触发 version bump）。"""
    from tcm_extract_engine.gepa.runner import (
        ledger_path_for,
        load_run_best_constraint,
        promote_constraint,
    )

    if not run_dir.is_dir():
        console.print(f"[red]错误:[/] 产物目录不存在: {run_dir}")
        raise typer.Exit(1)
    try:
        block, run_template = load_run_best_constraint(run_dir)
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc

    target = template or run_template
    if not block:
        console.print("[yellow]胜出约束块为空（等同基线），晋升将移除受管块（回滚）。[/]")
    _, oroot = _gepa_roots(None, out_root)
    try:
        res = promote_constraint(
            template=target,
            constraint_block=block,
            reviewer=reviewer,
            run_id=run_dir.name,
            prompts_root=prompts_root,
            dry_run=dry_run,
            ledger_path=ledger_path_for(oroot),
        )
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc

    tag = "【dry-run】" if res.dry_run else ""
    if not res.changed:
        console.print(f"[dim]{tag}模板无变化：{res.prompts_path}[/]")
        return
    console.print(
        f"[green]✓ {tag}已晋升[/] {res.template}  "
        f"version {res.old_version} → {res.new_version}"
        + ("（bump）" if res.version_bumped else "")
    )
    console.print(f"  路径: {res.prompts_path}")
    if not res.dry_run:
        console.print("  [yellow]下一步：对相关书重刷 extract（version 已变，幂等判定会重抽）：[/]")
        console.print("    tcm-engine book rerun <book_id> --from extract")


@gepa_app.command("runs")
def gepa_runs(
    out_root: Annotated[Path | None, typer.Option("--out-root", help="进化产物根目录")] = None,
    book_id: Annotated[str | None, typer.Option("--book", help="只看某本书")] = None,
) -> None:
    """列出已落盘的 evolve 运行产物。"""
    from tcm_extract_engine.core.artifacts import load_artifact

    _, oroot = _gepa_roots(None, out_root)
    if not oroot.is_dir():
        console.print("[dim]暂无进化产物[/]")
        return
    rows: list[tuple[str, str, str, str, str]] = []
    for run_json in sorted(oroot.glob("*/*/*/run.json")):
        try:
            data = load_artifact(run_json)
        except Exception:
            continue
        bid = data["header"]["book_id"]
        if book_id and bid != book_id:
            continue
        base = data.get("baseline_fitness")
        rows.append(
            (
                bid,
                str(data.get("template", "")),
                str(data.get("run_id", "")),
                f"{base:.3f}" if isinstance(base, float) else "-",
                f"{data.get('best_fitness', 0):.3f}",
            )
        )
    if not rows:
        console.print("[dim]暂无进化产物[/]")
        return
    table = Table(title="GEPA 进化产物")
    for col in ("Book ID", "模板", "run_id", "baseline", "best"):
        table.add_column(col, style="cyan")
    for r in rows:
        table.add_row(*r)
    console.print(table)


@gepa_app.command("evolve-multi")
def gepa_evolve_multi(
    book_ids: Annotated[list[str], typer.Argument(help="一个或多个书籍 ID（跨书 Pareto 进化）")],
    template: Annotated[
        str, typer.Option("--template", "-t", help="被进化的 A 模板文件名")
    ] = "general_rule.md",
    books_root: BooksRoot = None,
    out_root: Annotated[
        Path | None, typer.Option("--out-root", help="进化产物根目录（默认 workspace/gepa）")
    ] = None,
    population: Annotated[int, typer.Option("--population", help="种群大小")] = 2,
    generations: Annotated[int, typer.Option("--generations", help="最大代数")] = 2,
    budget: Annotated[int, typer.Option("--budget", help="评估调用预算上限")] = 6,
    verify_sample: Annotated[
        int, typer.Option("--verify-sample", help="B 核验采样上限；-1=全量")
    ] = 20,
    no_baseline: Annotated[bool, typer.Option("--no-baseline", help="跳过基线评估")] = False,
    merge: Annotated[
        bool, typer.Option("--merge", help="启用 system-aware 合并（官方 GEPA 机制④）")
    ] = False,
) -> None:
    """外圈飞轮：在多本书切片上跨书进化同一模板（Pareto 跨书选择），落盘 + 记台账。"""
    from tcm_extract_engine.gepa.flywheel import evolve_across_books

    br, oroot = _gepa_roots(books_root, out_root)

    def on_gen(rep: Any) -> None:
        console.print(
            f"  [dim]gen {rep.generation}[/] best={rep.best_fitness:.4f} avg={rep.avg_fitness:.4f}"
        )

    async def _run() -> Any:
        return await evolve_across_books(
            book_ids=list(book_ids),
            template=template,
            books_root=br,
            out_root=oroot,
            population=population,
            generations=generations,
            budget=budget,
            verify_sample=None if verify_sample < 0 else verify_sample,
            with_baseline=not no_baseline,
            use_merge=merge,
            on_generation=on_gen,
        )

    console.print(
        f"[bold]GEPA 跨书进化[/] books={len(book_ids)} template={template}"
        f"{' merge=on' if merge else ''}\n"
    )
    try:
        result = asyncio.run(_run())
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title=f"跨书进化结果 {result.run_id}")
    table.add_column("字段", style="cyan")
    table.add_column("值", style="green")
    base = f"{result.baseline_fitness:.4f}" if result.baseline_fitness is not None else "-"
    table.add_row("参与书数", str(len(result.book_id.split(","))))
    table.add_row("baseline(聚合)", base)
    table.add_row("best(聚合)", f"{result.best_fitness:.4f}")
    table.add_row("是否提升", "✓" if result.improved else "✗")
    table.add_row("轨迹", str([round(f, 4) for f in result.fitness_history]))
    table.add_row("评估次数", str(result.eval_calls))
    table.add_row("产物目录", str(result.run_dir))
    console.print(table)
    console.print("\n[bold]胜出约束块：[/]")
    console.print(result.best_constraint or "[dim]（空，等同基线）[/]")
    console.print(f"\n[yellow]晋升：[/] tcm-engine gepa promote {result.run_dir} --reviewer <you>")


@gepa_app.command("evolve-all")
def gepa_evolve_all(
    book_ids: Annotated[list[str], typer.Argument(help="一个或多个书籍 ID（跨书 Pareto 进化）")],
    books_root: BooksRoot = None,
    out_root: Annotated[
        Path | None, typer.Option("--out-root", help="进化产物根目录（默认 workspace/gepa）")
    ] = None,
    templates: Annotated[
        list[str] | None, typer.Option("--template", "-t", help="限定模板；默认 4 个 A 模板")
    ] = None,
    population: Annotated[int, typer.Option("--population", help="种群大小")] = 2,
    generations: Annotated[int, typer.Option("--generations", help="最大代数")] = 2,
    budget: Annotated[int, typer.Option("--budget", help="每模板评估调用预算上限")] = 6,
    verify_sample: Annotated[
        int, typer.Option("--verify-sample", help="B 核验采样上限；-1=全量")
    ] = 20,
    no_baseline: Annotated[bool, typer.Option("--no-baseline", help="跳过基线评估")] = False,
    merge: Annotated[
        bool, typer.Option("--merge", help="启用 system-aware 合并（官方 GEPA 机制④）")
    ] = False,
) -> None:
    """规模化批处理：对 4 个 A 模板依次跑跨书飞轮，汇总「模板 × baseline/best/是否建议晋升」。"""
    from tcm_extract_engine.gepa.flywheel import evolve_all_templates

    br, oroot = _gepa_roots(books_root, out_root)

    def on_template(tpl: str, res: Any, skip: str | None) -> None:
        if skip is not None:
            console.print(f"  [dim]{tpl}[/] → [yellow]跳过[/] {skip}")
        else:
            console.print(
                f"  [dim]{tpl}[/] → best={res.best_fitness:.4f} "
                f"{'✓' if res.improved else '✗'} run={res.run_id}"
            )

    async def _run() -> Any:
        return await evolve_all_templates(
            book_ids=list(book_ids),
            books_root=br,
            out_root=oroot,
            templates=list(templates) if templates else None,
            population=population,
            generations=generations,
            budget=budget,
            verify_sample=None if verify_sample < 0 else verify_sample,
            with_baseline=not no_baseline,
            use_merge=merge,
            on_template=on_template,
        )

    console.print(f"[bold]GEPA 全模板进化[/] books={len(book_ids)}{' merge=on' if merge else ''}\n")
    try:
        outcome = asyncio.run(_run())
    except Exception as exc:
        console.print(f"[red]✗[/] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title="全模板进化汇总")
    for col in ("模板", "baseline", "best", "↑", "建议", "run"):
        table.add_column(col, style="cyan")
    for res in outcome.results:
        base = f"{res.baseline_fitness:.3f}" if res.baseline_fitness is not None else "-"
        advise = f"promote {res.run_dir}" if res.improved else "维持基线"
        table.add_row(
            res.template,
            base,
            f"{res.best_fitness:.3f}",
            "✓" if res.improved else "✗",
            advise,
            res.run_id,
        )
    for tpl, reason in outcome.skipped.items():
        table.add_row(tpl, "-", "-", "", f"[yellow]跳过[/] {reason[:40]}", "-")
    console.print(table)
    console.print("\n[dim]晋升某模板：tcm-engine gepa promote <run_dir> --reviewer <you>[/]")


@gepa_app.command("trend")
def gepa_trend(
    out_root: Annotated[Path | None, typer.Option("--out-root", help="进化产物根目录")] = None,
    template: Annotated[str | None, typer.Option("--template", "-t", help="只看某模板")] = None,
    limit: Annotated[int, typer.Option("--limit", help="最多显示条数")] = 30,
) -> None:
    """质量趋势看板：按时间线展示 evolve/promote 的适应度与晋升轨迹。"""
    from tcm_extract_engine.gepa.runner import ledger_path_for, load_trend

    _, oroot = _gepa_roots(None, out_root)
    records = load_trend(ledger_path_for(oroot))
    if template:
        records = [r for r in records if r.template == template]
    if not records:
        console.print("[dim]暂无趋势记录[/]")
        return
    records = records[-limit:]

    table = Table(title=f"GEPA 质量趋势（{oroot / '_trend.jsonl'}）")
    for col in ("时间", "类型", "模板", "书数", "baseline", "best", "↑", "晋升/版本"):
        table.add_column(col, style="cyan")
    for r in records:
        base = f"{r.baseline_fitness:.3f}" if r.baseline_fitness is not None else "-"
        best = f"{r.best_fitness:.3f}" if r.best_fitness is not None else "-"
        if r.kind == "promote":
            tail = f"{r.old_version}→{r.new_version}" if r.promoted else "回滚"
            base = best = "-"
        else:
            tail = f"calls={r.eval_calls} tok={r.total_tokens}"
        table.add_row(
            r.ts[:19],
            r.kind,
            r.template,
            str(len(r.book_ids)),
            base,
            best,
            "✓" if r.improved else "",
            tail,
        )
    console.print(table)
