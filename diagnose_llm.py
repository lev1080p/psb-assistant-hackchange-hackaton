"""Explicit, synthetic-only LLM smoke test. No network activity on import."""
import argparse
import asyncio
import logging
import time
from pydantic import ValidationError
from config import Settings
from dependencies import BackendUnavailable, LanguageModel, UpstreamError, UpstreamTimeout
from llm_client import OpenAICompatibleClient
from prompts import answer_messages, verification_messages
from schemas import ChatRequest, Draft, Fragment, SupportCheck
from verification import validate_citations


async def diagnose(client: LanguageModel, settings: Settings) -> list[dict]:
    """Perform three synthetic calls; report no model content or exception text."""
    source = Fragment(fragment_id="diagnostic-f", document_id="diagnostic-d",
                      document_title="Synthetic marker", text="Marker ALPHA")
    request = ChatRequest(question="Повтори маркер из предоставленной тестовой записи.")
    draft = None
    reports = []
    for stage in ("text", "draft", "support"):
        started = time.perf_counter()
        try:
            if stage == "text":
                messages = [{"role": "user", "content": "Reply with the word READY."}]
            elif stage == "draft":
                messages = answer_messages(request, [source])
            else:
                if draft is None:
                    reports.append({"stage": stage, "status": "skipped", "seconds": 0.0})
                    continue
                messages = verification_messages(request, draft, [source])
            async with asyncio.timeout(settings.request_timeout):
                content = await client.complete(messages)
            if stage == "text":
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty text")
            elif stage == "draft":
                candidate = Draft.model_validate_json(content)
                if candidate.status != "answer" or not validate_citations(candidate, [source]):
                    raise ValueError("Unsupported synthetic draft")
                draft = candidate
            else:
                if not SupportCheck.model_validate_json(content).supported:
                    raise ValueError("Support check rejected synthetic draft")
            status = "ok"
        except BackendUnavailable:
            status = "configuration_or_authorization_error"
        except (UpstreamTimeout, TimeoutError):
            status = "timeout"
        except UpstreamError:
            status = "upstream_error"
        except (ValidationError, ValueError):
            status = "invalid_model_output"
        except Exception:
            status = "unexpected_error"
        reports.append({"stage": stage, "status": status,
                        "seconds": round(time.perf_counter() - started, 3)})
        if status == "configuration_or_authorization_error":
            break
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic LLM checks; --run explicitly enables network calls.")
    parser.add_argument("--run", action="store_true", help="Send synthetic requests using environment settings")
    args = parser.parse_args(argv)
    if not args.run:
        print("No requests sent. Use --run to explicitly enable LLM diagnostics.")
        return 0
    # Do not expose provider URLs, bodies or exception chains through library logs.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        settings = Settings.from_env()
        settings.validate_llm()
        reports = asyncio.run(diagnose(OpenAICompatibleClient(settings), settings))
        for report in reports:
            print(f"{report['stage']}: {report['status']} ({report['seconds']:.3f}s)")
        return 0 if len(reports) == 3 and all(r['status'] == 'ok' for r in reports) else 1
    except KeyboardInterrupt:
        print("Diagnostics cancelled.")
        return 130
    except Exception:
        print("Diagnostics failed: check environment configuration. Details suppressed.")
        return 1
    finally:
        logging.disable(previous_disable)


if __name__ == "__main__":
    raise SystemExit(main())
