"""Command-line interface for RAG system.

Provides easy-to-use commands for document processing and querying.
Optimized for 612MB dataset with progress tracking.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from ..core.pipeline import RAGPipeline
from ..config.settings import settings

console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
def cli(verbose: bool, config: Optional[str]):
    """RAG System CLI - Process documents and answer questions."""
    setup_logging(verbose)
    
    if config:
        # Load configuration file if provided
        console.print(f"[yellow]Loading configuration from: {config}[/yellow]")


@cli.command()
@click.argument('documents_path', type=click.Path(exists=True, path_type=Path))
@click.option('--collection', '-c', default=None, help='Collection name for vector store')
@click.option('--recursive/--no-recursive', default=True, help='Process subdirectories')
@click.option('--force-reindex', is_flag=True, help='Force re-indexing of existing documents')
def ingest(documents_path: Path, collection: Optional[str], recursive: bool, force_reindex: bool):
    """Ingest documents into the RAG system."""
    console.print(Panel.fit(
        f"[bold blue]Ingesting documents from:[/bold blue] {documents_path}",
        title="RAG Document Ingestion"
    ))
    
    async def _ingest():
        try:
            # Create pipeline
            pipeline = RAGPipeline(collection_name=collection)
            
            console.print("[yellow]Initializing RAG pipeline...[/yellow]")
            await pipeline.initialize(skip_generator=True)
            
            console.print(f"[yellow]Processing documents (recursive={recursive})...[/yellow]")
            
            # Process documents with progress tracking
            stats = await pipeline.process_documents(
                documents_path=documents_path,
                recursive=recursive, 
                force_reindex=force_reindex
            )
            
            # Display results
            results_table = Table(title="Processing Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")
            
            results_table.add_row("Total Documents", str(stats.total_documents))
            results_table.add_row("Processed Documents", str(stats.processed_documents))
            results_table.add_row("Total Chunks", str(stats.total_chunks))
            results_table.add_row("Success Rate", f"{stats.success_rate:.1%}")
            results_table.add_row("Processing Time", f"{stats.processing_time:.2f}s")
            
            if stats.errors:
                results_table.add_row("Errors", str(len(stats.errors)))
            
            console.print(results_table)
            
            # Show pipeline stats
            pipeline_stats = pipeline.get_pipeline_stats()
            console.print(f"\n[green]✓ Successfully processed {stats.processed_documents} documents!")
            console.print(f"[green]✓ Created {stats.total_chunks} chunks in vector store")
            console.print(f"[green]✓ Collection: {collection or settings.collection_name}")
            
        except Exception as e:
            console.print(f"[red]✗ Error during ingestion: {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_ingest())


@cli.command()
@click.argument('question', type=str)
@click.option('--collection', '-c', default=None, help='Collection name to query')
@click.option('--top-k', '-k', default=5, help='Number of chunks to retrieve')
@click.option('--method', '-m', 
              type=click.Choice(['hybrid', 'vector', 'bm25']), 
              default='hybrid', help='Retrieval method')
@click.option('--template', '-t',
              type=click.Choice(['default', 'citation', 'summary', 'comparison']),
              default='default', help='Response template type')
@click.option('--show-sources', is_flag=True, help='Show source information')
@click.option('--show-chunks', is_flag=True, help='Show retrieved chunks')
def query(question: str, collection: Optional[str], top_k: int, method: str, 
          template: str, show_sources: bool, show_chunks: bool):
    """Query the RAG system with a question."""
    console.print(Panel.fit(
        f"[bold blue]Question:[/bold blue] {question}",
        title="RAG Query"
    ))
    
    async def _query():
        try:
            # Create pipeline
            pipeline = RAGPipeline(collection_name=collection)
            
            console.print("[yellow]Initializing RAG pipeline...[/yellow]")
            await pipeline.initialize()
            
            # Check if documents are indexed
            docs_info = pipeline.get_indexed_documents_info()
            if not docs_info:
                console.print("[red]✗ No documents found in the collection. Run 'ingest' first.[/red]")
                return
            
            console.print(f"[yellow]Querying {len(docs_info)} indexed documents...[/yellow]")
            
            # Process query
            response = await pipeline.query(
                question=question,
                top_k=top_k,
                retrieval_method=method,
                template_type=template
            )
            
            # Display answer
            answer_panel = Panel(
                Markdown(response.answer),
                title=f"Answer (Confidence: {response.confidence_score:.1%})",
                title_align="left"
            )
            console.print(answer_panel)
            
            # Display metadata
            metadata_table = Table(show_header=False, box=None)
            metadata_table.add_column("Key", style="cyan")
            metadata_table.add_column("Value", style="white")
            
            metadata_table.add_row("Response Time", f"{response.response_time:.3f}s")
            metadata_table.add_row("Retrieval Method", method)
            metadata_table.add_row("Sources Found", str(len(response.sources)))
            
            console.print(metadata_table)
            
            # Show sources if requested
            if show_sources and response.sources:
                sources_table = Table(title="Sources")
                sources_table.add_column("File", style="cyan")
                sources_table.add_column("Chunk ID", style="yellow")
                
                for source in response.sources[:5]:  # Show top 5 sources
                    sources_table.add_row(
                        source.get('filename', 'Unknown'),
                        source.get('chunk_id', 'Unknown')
                    )
                
                console.print(sources_table)
            
            # Show retrieved chunks if requested
            if show_chunks and response.retrieval_result:
                console.print("\n[bold]Retrieved Chunks:[/bold]")
                for i, (chunk, score) in enumerate(zip(
                    response.retrieval_result.chunks[:3], 
                    response.retrieval_result.scores[:3]
                ), 1):
                    chunk_panel = Panel(
                        chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        title=f"Chunk {i} (Score: {score:.3f})",
                        title_align="left"
                    )
                    console.print(chunk_panel)
            
        except Exception as e:
            console.print(f"[red]✗ Error during query: {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_query())


@cli.command()
@click.option('--collection', '-c', default=None, help='Collection name to check')
def status(collection: Optional[str]):
    """Show RAG system status and statistics."""
    async def _status():
        try:
            # Create pipeline
            pipeline = RAGPipeline(collection_name=collection)
            await pipeline.initialize()
            
            # Get health check
            health = await pipeline.health_check()
            
            # Display health status
            status_color = "green" if health['status'] == 'healthy' else "red"
            console.print(f"\n[bold {status_color}]System Status: {health['status'].upper()}[/bold {status_color}]")
            
            # Component status
            if 'components' in health:
                components_table = Table(title="Component Status")
                components_table.add_column("Component", style="cyan")
                components_table.add_column("Status", style="white")
                
                for component, status in health['components'].items():
                    color = "green" if status == "healthy" else "red"
                    components_table.add_row(component, f"[{color}]{status}[/{color}]")
                
                console.print(components_table)
            
            # Get pipeline stats
            stats = pipeline.get_pipeline_stats()
            
            # Display statistics
            stats_table = Table(title="Pipeline Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            stats_table.add_row("Documents Processed", str(stats.get('documents_processed', 0)))
            stats_table.add_row("Chunks Indexed", str(stats.get('chunks_indexed', 0)))
            stats_table.add_row("Queries Processed", str(stats.get('queries_processed', 0)))
            
            if stats.get('avg_query_time'):
                stats_table.add_row("Avg Query Time", f"{stats['avg_query_time']:.3f}s")
            
            console.print(stats_table)
            
            # Document information
            docs_info = pipeline.get_indexed_documents_info()
            if docs_info:
                docs_table = Table(title="Indexed Documents")
                docs_table.add_column("Filename", style="cyan")
                docs_table.add_column("Chunks", style="yellow")
                docs_table.add_column("Size (chars)", style="green")
                
                for doc in docs_info[:10]:  # Show first 10 documents
                    docs_table.add_row(
                        doc['filename'],
                        str(doc['chunk_count']),
                        f"{doc['total_chars']:,}"
                    )
                
                if len(docs_info) > 10:
                    docs_table.add_row("...", "...", "...")
                    docs_table.add_row(f"({len(docs_info)} total documents)", "", "")
                
                console.print(docs_table)
            else:
                console.print("[yellow]No documents indexed yet. Run 'ingest' to add documents.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]✗ Error getting status: {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_status())


@cli.command()
@click.option('--collection', '-c', default=None, help='Collection name to reset')
@click.confirmation_option(prompt='Are you sure you want to reset all data?')
def reset(collection: Optional[str]):
    """Reset the RAG system (delete all indexed data)."""
    async def _reset():
        try:
            pipeline = RAGPipeline(collection_name=collection)
            await pipeline.initialize()
            
            console.print("[yellow]Resetting RAG system...[/yellow]")
            await pipeline.reset()
            
            console.print("[green]✓ RAG system reset successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error during reset: {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_reset())


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
def serve(host: str, port: int, workers: int, reload: bool, log_level: str):
    """Start the FastAPI server."""
    console.print(Panel.fit(
        f"[bold blue]Starting RAG API Server[/bold blue]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Workers: {workers}\n"
        f"Reload: {reload}",
        title="Server Configuration"
    ))
    
    try:
        from .server import start_server
        start_server(
            host=host,
            port=port,
            reload=reload,
            workers=workers
        )
    except ImportError as e:
        console.print(f"[red]✗ Failed to import server: {e}")
        console.print("[yellow]Make sure FastAPI and uvicorn are installed:[/yellow]")
        console.print("pip install fastapi uvicorn[standard]")
        raise click.ClickException("Server dependencies not found")
    except Exception as e:
        console.print(f"[red]✗ Failed to start server: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API server URL')
@click.argument('query_text', type=str)
def api_query(url: str, query_text: str):
    """Query the RAG system via API."""
    console.print(Panel.fit(
        f"[bold blue]API Query:[/bold blue] {query_text}",
        title="RAG API Query"
    ))
    
    async def _api_query():
        try:
            from .client import RAGClient
            
            async with RAGClient(base_url=url) as client:
                # Check health first
                try:
                    health = await client.health_check()
                    console.print(f"[green]✓ API server is {health['status']}[/green]")
                except Exception as e:
                    console.print(f"[red]✗ API server not accessible: {e}[/red]")
                    return
                
                # Execute query
                result = await client.query(query_text)
                
                # Display results
                answer_panel = Panel(
                    Markdown(result['answer']),
                    title=f"Answer (Confidence: {result['confidence_score']:.1%})",
                    title_align="left"
                )
                console.print(answer_panel)
                
                # Display metadata
                metadata_table = Table(show_header=False, box=None)
                metadata_table.add_column("Key", style="cyan")
                metadata_table.add_column("Value", style="white")
                
                metadata_table.add_row("Response Time", f"{result['response_time']:.3f}s")
                metadata_table.add_row("Sources Found", str(len(result['sources'])))
                
                console.print(metadata_table)
                
        except ImportError as e:
            console.print(f"[red]✗ Failed to import API client: {e}")
            console.print("[yellow]Make sure httpx is installed:[/yellow]")
            console.print("pip install httpx")
            raise click.ClickException("API client dependencies not found")
        except Exception as e:
            console.print(f"[red]✗ API query failed: {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_api_query())


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API server URL')
def api_status(url: str):
    """Check API server status."""
    async def _api_status():
        try:
            from .client import RAGClient
            
            async with RAGClient(base_url=url) as client:
                # Get health and stats
                health = await client.health_check()
                stats = await client.get_stats()
                
                # Display health
                status_color = "green" if health['status'] == 'healthy' else "red"
                console.print(f"\n[bold {status_color}]API Status: {health['status'].upper()}[/bold {status_color}]")
                
                # Display server info
                server_table = Table(title="Server Information")
                server_table.add_column("Property", style="cyan")
                server_table.add_column("Value", style="green")
                
                server_table.add_row("URL", url)
                server_table.add_row("Version", health.get('version', 'Unknown'))
                server_table.add_row("Timestamp", health.get('timestamp', 'Unknown'))
                
                console.print(server_table)
                
                # Display statistics
                if 'vector_store' in stats:
                    vs_stats = stats['vector_store']
                    vs_table = Table(title="Vector Store Statistics")
                    vs_table.add_column("Metric", style="cyan")
                    vs_table.add_column("Value", style="green")
                    
                    vs_table.add_row("Total Chunks", str(vs_stats.get('total_chunks', 0)))
                    vs_table.add_row("Collection", vs_stats.get('collection_name', 'Unknown'))
                    vs_table.add_row("Embedding Model", vs_stats.get('embedding_model', 'Unknown'))
                    
                    console.print(vs_table)
                
        except ImportError as e:
            console.print(f"[red]✗ Failed to import API client: {e}")
            raise click.ClickException("API client dependencies not found")
        except Exception as e:
            console.print(f"[red]✗ Failed to connect to API: {e}")
            raise click.ClickException(str(e))
    
    asyncio.run(_api_status())


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()